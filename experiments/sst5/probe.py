"""Probe a labeler on dev (per-item, guidance v1 + 20 seeds, soft). usage: probe.py MODEL REASONING(off|low|medium|high) SUBSET(B|all) [MAXTOK]"""
import asyncio, sys, json
sys.path.insert(0, "experiments/sst5")
from lab import *
from ownyourjev.llm import OpenRouterClient
model, reas, subset = sys.argv[1], sys.argv[2], sys.argv[3]
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else (200 if reas == "off" else 4000)
g = open("experiments/sst5/guidance_v1.txt").read()
exs = DEV[100:] if subset == "B" else DEV
eb = {"provider": {"sort": "throughput"}}
if reas != "off": eb["reasoning"] = {"enabled": True, "effort": reas}
name = f"probe_{model.split('/')[1]}_{reas}_{subset}"
async def main():
    c = OpenRouterClient(model=model, extra_body=eb, concurrency=int(__import__("os").environ.get("CONC","16")))
    res = await label_examples(c, Q, exs, purpose=f"sst5:{name}", seeds=SEEDS, guidance=g, distribution=True, max_tokens=maxtok, progress=False)
    await c.aclose(); save(name, res)
    print(name, "cost", round(c.stats["cost"], 4), "calls", c.stats["calls"], "fails", sum(r.labels is None for r in res), "out_tok", c.stats["completion_tokens"])
asyncio.run(main())
uids = [e.uid for e in exs]; y = np.array([e.labels["answer"] for e in exs])
p = probs_of(load(name), uids); base = probs_of(load("dev_v1_deepseek-v4.1-flash"), uids)
a, b = p.argmax(1) == y, base.argmax(1) == y
rng = np.random.default_rng(0); d = [a[i].mean() - b[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(5000))]
print(f"{name}: acc {a.mean():.3f} vs flash {b.mean():.3f} diff {a.mean()-b.mean():+.3f} CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] spearman {stats.spearmanr(p@np.arange(5), y).statistic:.3f} dist {np.bincount(p.argmax(1), minlength=5)}")
