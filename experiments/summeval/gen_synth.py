"""Generate synthetic summaries of deliberately varied quality per article (+ optional new LLM-written articles).
usage: gen_synth.py summaries <tag> <n_calls_per_article> [--articles real|new]
       gen_synth.py articles <n>
"""
import asyncio, json, random, re, sys
from common import *
from ownyourjev.llm import OpenRouterClient
from ownyourjev.data import Example, write_jsonl, read_jsonl

GEN_MODEL = "qwen/qwen3.7-flash"

PROFILES = {
    "excellent": "an excellent abstractive summary: covers the central event and the 2-3 most important facts, well organized, fluent, fully faithful",
    "good_extractive": "copy the 3 most important sentences of the article verbatim (usually from the beginning), in article order",
    "lead_extractive": "copy the first 3 sentences of the article verbatim, even if one is a minor detail",
    "random_extractive": "copy 3 sentences from the middle or end of the article verbatim (peripheral details, not the main point), in random order",
    "minor_hallucination": "a fluent, well-organized summary that is faithful except for ONE wrong detail (a changed number, date, name or place)",
    "major_hallucination": "a fluent summary that invents several facts not in the article (fake quotes, wrong outcome, invented people or numbers)",
    "misattribution": "a summary that attributes a quote or action to the wrong person or organization from the article",
    "disfluent": "a summary with poor grammar: missing words, wrong verb forms, a duplicated word, awkward word choices",
    "garbled": "a summary written like a weak neural summarizer: repetition of phrases, a fused run-on sentence, a sentence that trails off",
    "incoherent_order": "a summary whose sentences are important but in an illogical order: starts with a pronoun or detail with no antecedent, jumps between topics",
    "redundant": "a summary that states the main fact two or three times in slightly different words and says little else",
    "truncated": "a decent summary whose last sentence is cut off in the middle (ends abruptly without punctuation)",
    "off_focus": "a fluent summary that focuses on a minor side detail of the article and misses its main point",
    "verbose": "a long summary (5-6 sentences) packed with unimportant details, quotes and background, but faithful",
    "too_short": "a one-sentence summary that is faithful but leaves out most key facts",
    "mixed_flaws": "a summary with a moderate amount of several flaws: one unsupported detail, one awkward sentence, slightly disorganized",
    "headline_bullets": "3-4 short highlight-style sentences like CNN/DailyMail bullet highlights, faithful and important",
    "paraphrase_drift": "a paraphrase of the key points that subtly changes meaning (e.g., exaggerates, reverses a cause, or overgeneralizes)",
}


def ptb_style(s: str, rng: random.Random) -> str:
    """Mimic the CNN/DM dataset display style: tokenized, often lower-cased."""
    s = s.replace("(", " -lrb- ").replace(")", " -rrb- ")
    s = re.sub(r'"([^"]*)"', r"`` \1 ''", s)
    s = re.sub(r"(\w)'(s|re|ve|ll|d|m)\b", r"\1 '\2", s)
    s = re.sub(r"n't\b", " n't", s)
    s = re.sub(r"([,.;:!?])(\s|$)", r" \1\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if rng.random() < 0.85 else s


def build_msgs(article: str, profiles: list[str]):
    spec = "\n".join(f'{i + 1}. "{p}": {PROFILES[p]}' for i, p in enumerate(profiles))
    sys_ = ("You create training data for an automatic summary-quality evaluator. Given a news article, you write summaries "
            "that each follow an exact quality profile, so the evaluator learns to tell good from bad summaries. "
            "Summaries are 2-4 sentences (about 40-90 words) unless the profile says otherwise. Follow the profile exactly, "
            "including its flaws. Reply with JSON only.")
    user = (f"ARTICLE:\n{article}\n\nWrite one summary for each profile:\n{spec}\n\n"
            'Return {"summaries": [{"profile": "<profile name>", "summary": "<text>"}, ...]} in the same order.')
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def real_articles():
    """(group, article, fold) for the 86 eval documents (fold = eval fold) and the 14 carved documents (fold = -1)."""
    ev = P.eval_inputs(T)
    folds = P.eval_folds(T)
    fold_of = {}
    for k, (_, te) in enumerate(folds):
        for i in te:
            fold_of[ev[i].group] = k
    arts = {}
    for e in ev:
        arts.setdefault(e.group, (e.state["source_document"], fold_of[e.group]))
    for e in P.seeds(T) + P.dev(T):
        arts.setdefault(e.group, (e.state["source_document"], -1))
    return [(g, a, f) for g, (a, f) in sorted(arts.items())]


async def gen_summaries(tag: str, n_calls: int, which: str):
    rng = random.Random(f"synth-{tag}")
    cl = OpenRouterClient(model=GEN_MODEL, concurrency=32, temperature=0.9)
    if which == "real":
        arts = real_articles()
    else:
        arts = [(e.group, e.state["source_document"], -2) for e in read_jsonl(DATA / "new_articles.jsonl")]
    names = list(PROFILES)
    jobs = []
    for g, a, f in arts:
        for c in range(n_calls):
            prof = rng.sample(names, 9)
            jobs.append((g, a, f, c, prof))

    async def one(job):
        g, a, f, c, prof = job
        try:
            raw = await cl.complete_json(build_msgs(a, prof), purpose=f"{T}:gen:{tag}", max_tokens=1800)
        except Exception as e:  # noqa
            return []
        items = raw.get("summaries") if isinstance(raw, dict) else None
        return [(g, a, f, c, it) for it in (items or []) if isinstance(it, dict) and isinstance(it.get("summary"), str)]

    res = await cl.map(jobs, one, desc=f"gen:{tag}")
    out = []
    for batch in res:
        for j, (g, a, f, c, it) in enumerate(batch):
            s = it["summary"].strip()
            if len(s) < 20:
                continue
            r = random.Random(f"{g}-{c}-{j}")
            style = "ptb" if r.random() < 0.65 else "raw"
            if style == "ptb":
                s = ptb_style(s, r)
            out.append(Example(f"syn-{tag}-{g}-{c}-{j}", {"source_document": a, "summary": s}, None, None, g,
                               {"synthetic": True, "profile": it.get("profile"), "fold": f, "style": style}))
    write_jsonl(DATA / f"synth_{tag}.jsonl", out)
    print(len(out), "summaries", f"cost={cl.stats['cost']:.4f}")


TOPICS = ["premier league football transfer", "celebrity wedding", "US crime and court case", "UK politics", "royal family",
          "medical study", "airline incident", "animal rescue at a zoo", "tennis tournament", "military incident",
          "tech company product", "weather disaster", "school scandal", "Middle East conflict", "space mission",
          "celebrity fashion", "murder trial", "police shooting", "NFL or NBA news", "cycling or athletics record",
          "auction of memorabilia", "travel and tourism", "diet and health trend", "immigration", "terror attack aftermath",
          "business merger", "TV show or film", "viral social media story", "plane crash investigation", "cricket or rugby"]


async def gen_articles(n: int):
    rng = random.Random("articles")
    cl = OpenRouterClient(model=GEN_MODEL, concurrency=32, temperature=1.0)
    src = [a for _, a, _ in real_articles()]
    jobs = []
    for i in range(n):
        jobs.append((i, rng.choice(TOPICS), rng.choice(["CNN", "Daily Mail"]), rng.choice([350, 450, 550, 650])))

    async def one(job):
        i, topic, outlet, words = job
        msgs = [{"role": "system", "content": "You write realistic, fictional news articles in the exact style of CNN/DailyMail online news (2015). Plain text only, no headline, no markdown. Reply with JSON only."},
                {"role": "user", "content": f"Write a fictional {outlet}-style news article (about {words} words) on this topic: {topic}. Use specific (fictional) names, places, numbers, dates and several quotes; include background paragraphs and minor details like real articles do. Article #{i}.\nReturn {{\"article\": \"<text>\"}}"}]
        try:
            raw = await cl.complete_json(msgs, purpose=f"{T}:gen_articles", max_tokens=1400)
            a = raw.get("article") if isinstance(raw, dict) else None
        except Exception:
            a = None
        return (i, a)

    res = await cl.map(jobs, one, desc="articles")
    out = [Example(f"newart-{i}", {"source_document": a.strip(), "summary": ""}, None, None, f"newart-{i}", {"synthetic_article": True})
           for i, a in res if isinstance(a, str) and len(a) > 800]
    write_jsonl(DATA / "new_articles.jsonl", out)
    print(len(out), "articles", f"cost={cl.stats['cost']:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "summaries":
        asyncio.run(gen_summaries(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "real"))
    else:
        asyncio.run(gen_articles(int(sys.argv[2])))
