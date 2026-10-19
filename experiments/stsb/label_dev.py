"""Label dev (or pool) with a given labeler config; report Spearman (expected level vs gold) on dev halves."""
import asyncio, argparse, json
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient

GUIDE = {
"none": None,
"semeval": """Follow the SemEval STS annotation guidelines (scale 0-5). Judge MEANING, not word overlap.
- 5: completely equivalent; paraphrases. Minor wording, tense, punctuation, or synonym differences do not lower the score.
- 4: mostly equivalent; only unimportant details differ (e.g. an adjective, a minor modifier, 'a man' vs 'a young man', a slightly different number in a headline that otherwise matches).
- 3: roughly equivalent, but some important information differs or is missing (e.g. one sentence adds a substantial clause; different counts; a different but related object).
- 2: not equivalent, but share some details (same kind of event or same actor, but different key elements, e.g. different subject or different action).
- 1: not equivalent, but on the same topic (same domain/theme only, e.g. both about stock markets in different countries, both about arrests).
- 0: completely dissimilar, different topics.
Important conventions of this dataset:
- Different named entities (countries, people, companies) or different numbers in otherwise parallel news headlines usually put the pair at 1-2, not 3+, because the main fact differs.
- A sentence that is a strict subset of the other (one adds details) is typically 3-4 depending on how important the added information is.
- Negation or opposite meaning lowers the score strongly even with high word overlap.
- Use the whole scale; the human scores are spread roughly evenly over 0-5 with a mean near 2.75.""",
}
GUIDE["semeval_soft"] = GUIDE["semeval"] + """
When you give probabilities, they express where the averaged score of several human annotators would fall: put mass on adjacent levels when the pair is between two levels, so that the expected level matches your best real-valued estimate of the human score."""

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--guide", default="none")
    ap.add_argument("--dist", action="store_true")
    ap.add_argument("--seeds", action="store_true")
    ap.add_argument("--sample", type=int, default=-1, help="sampling seed (temperature 1)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--maxp", type=float, default=0.0)
    a = ap.parse_args()
    eb = {}
    if a.sample >= 0: eb["seed"] = a.sample
    if a.maxp: eb["provider"] = {"sort": "throughput", "max_price": {"prompt": a.maxp, "completion": a.maxp * 8}}
    cl = OpenRouterClient(model=a.model, temperature=1.0 if a.sample >= 0 else 0.0, extra_body=eb or None, concurrency=32)
    if a.split == "dev":
        exs = protocol.dev(T)
    else:
        exs = protocol.pool(T)
    seeds = protocol.seeds(T) if a.seeds else None
    if seeds:  # seeds' gold is real-valued: present it rounded as the level index
        from dataclasses import replace
        seeds = [replace(e, labels={"answer": int(round(e.gold["answer"]))}) for e in seeds]
    from dataclasses import replace
    inp = [replace(e, labels=None) for e in exs]
    out = await label_examples(cl, Q, inp, purpose=f"stsb:{a.tag}:{a.split}", seeds=seeds, guidance=GUIDE[a.guide], distribution=a.dist)
    save_jsonl(D / f"labels_{a.split}_{a.tag}.jsonl", out)
    print(a.tag, "stats", cl.stats)
    if a.split == "dev":
        g = np.array([e.gold["answer"] for e in exs]); s = np.array([expected(e) for e in out])
        ok = ~np.isnan(s)
        h = np.arange(len(g)) % 2 == 0
        print(f"{a.tag}: n_ok={ok.sum()} spearman all={sp(g[ok], s[ok]):.4f} A={sp(g[ok&h], s[ok&h]):.4f} B={sp(g[ok&~h], s[ok&~h]):.4f} pearson={np.corrcoef(g[ok], s[ok])[0,1]:.4f}")
if __name__ == "__main__":
    asyncio.run(main())
