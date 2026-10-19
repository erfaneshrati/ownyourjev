"""Pilot: natural (not deliberately flawed) summaries of real eval articles from several cheap LLMs and prompt styles,
mimicking the variety of real summarisation systems. Only articles from folds 1-4 (fold 0 stays held out for dev selection).
usage: gen_natural.py <n_articles>"""
import asyncio, random, sys
from common import *
from gen_synth import real_articles, ptb_style
from ownyourjev.llm import OpenRouterClient
from ownyourjev.data import Example, write_jsonl

MODELS = ["qwen/qwen3.7-flash", "google/gemini-2.5-flash-lite", "deepseek/deepseek-v4.1-flash"]
PROMPTS = {
    "three_sent": "Summarize the article in 3 sentences.",
    "highlights": "Write 3-4 short highlight sentences for this article, like the bullet highlights under a CNN or Daily Mail headline (plain sentences, no bullet symbols).",
    "one_para": "Write a one-paragraph summary of about 50 words.",
    "quick": "In a hurry, write a short summary of this article in 2-3 sentences.",
}

async def main(n):
    arts = [a for a in real_articles() if a[2] in (1, 2, 3, 4)]
    rng = random.Random("natural"); rng.shuffle(arts); arts = arts[:n]
    if len(sys.argv) > 2 and sys.argv[2] == "rest":  # every remaining eval article (all folds) not in the pilot
        done = {a[0] for a in arts}
        arts = [a for a in real_articles() if a[2] >= 0 and a[0] not in done]
    clients = {m: OpenRouterClient(model=m, concurrency=24, temperature=1.0) for m in MODELS}
    jobs = [(g, a, f, m, p) for g, a, f in arts for m in MODELS for p in PROMPTS]
    async def one(j):
        g, a, f, m, p = j
        msgs = [{"role": "system", "content": "You summarize news articles. Reply with JSON only."},
                {"role": "user", "content": f"ARTICLE:\n{a}\n\n{PROMPTS[p]}\nReturn {{\"summary\": \"<text>\"}}"}]
        try:
            raw = await clients[m].complete_json(msgs, purpose=f"{T}:gen_natural", max_tokens=300)
            s = raw.get("summary") if isinstance(raw, dict) else None
            return (j, s if isinstance(s, str) and len(s) > 20 else None)
        except Exception:
            return (j, None)
    res = await clients[MODELS[0]].map(jobs, one, desc="natural", progress=False)
    out = []
    for (g, a, f, m, p), s in res:
        if not s: continue
        r = random.Random(f"{g}-{m}-{p}")
        s = ptb_style(s, r) if r.random() < 0.65 else s.strip()
        out.append(Example(f"nat-{g}-{m.split('/')[-1]}-{p}", {"source_document": a, "summary": s}, None, None, g,
                           {"synthetic": True, "profile": f"nat_{p}", "gen_model": m, "fold": f}))
    write_jsonl(DATA / ("synth_natural2.jsonl" if len(sys.argv) > 2 else "synth_natural.jsonl"), out)
    print(len(out), "summaries", {m: round(c.stats["cost"], 4) for m, c in clients.items()})

if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
