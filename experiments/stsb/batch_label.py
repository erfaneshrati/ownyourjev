"""Batched labeling (N pairs per call, compact output) to cut prompt cost; same question + guidance as label_dev.py.
Output per item: a 6-number probability list (fmt=dist) or a real-valued 0-5 score (fmt=score).
Stored like label_examples output: labels.answer = argmax/rounded level, meta.soft.answer = distribution."""
import asyncio, argparse, json
from dataclasses import replace
from common import *
from label_dev import GUIDE
from ownyourjev.llm import OpenRouterClient
from ownyourjev.label import SYSTEM_PREAMBLE

FMT = {
"dist": 'Return one JSON object mapping each pair number (as a string) to a list of 6 probabilities [p0, p1, p2, p3, p4, p5] over the levels 0-5, each with at most 2 decimals, summing to 1. Example: {"1": [0, 0, 0.1, 0.6, 0.3, 0], "2": [...]}',
"score": 'Return one JSON object mapping each pair number (as a string) to your estimate of the averaged human similarity score, a real number from 0.0 to 5.0 with one decimal. Example: {"1": 3.4, "2": 0.8}',
}

def twohot(x, K=6):
    x = min(max(float(x), 0.0), K - 1.0)
    v = np.zeros(K); lo = int(np.floor(x)); hi = min(lo + 1, K - 1); w = x - lo
    v[lo] += 1 - w; v[hi] += w; return v

def system(guide, fmt, seeds=False):
    parts = [SYSTEM_PREAMBLE, "", "## Question", q.describe().replace("Ordered levels (answer with the level index as an integer):", "Ordered levels:")]
    if GUIDE[guide]:
        parts.append("\n## Labeling guidelines\n" + GUIDE[guide].strip())
    parts.append("\n## Task\nYou will receive several numbered sentence pairs. Judge each pair independently.\n## Output format\n" + FMT[fmt])
    if seeds:
        parts.append("\n## Examples (pair -> averaged human score)")
        for e in protocol.seeds(T):
            parts.append(f"sentence_1: {e.state['sentence_1'].strip()}\nsentence_2: {e.state['sentence_2'].strip()}\nscore: {e.gold['answer']:.1f}\n")
    return "\n".join(parts)

def user(batch):
    lines = []
    for i, e in enumerate(batch, 1):
        lines.append(f"### Pair {i}\nsentence_1: {e.state['sentence_1'].strip()}\nsentence_2: {e.state['sentence_2'].strip()}\n")
    return "\n".join(lines) + "\n## ANSWER (JSON only)"

async def label_batched(cl, exs, *, guide, fmt, bs, purpose, seeds=False):
    sysm = system(guide, fmt, seeds)
    batches = [exs[i:i + bs] for i in range(0, len(exs), bs)]
    async def one(batch):
        err = None
        for attempt in range(2):
            msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": user(batch) + (f"\n\nYour previous answer was invalid: {err}. Answer for ALL {len(batch)} pairs." if err else "")}]
            try:
                raw = await cl.complete_json(msgs, purpose=purpose, max_tokens=(40 if fmt == "dist" else 12) * len(batch) + 60)
                out = []
                for i, e in enumerate(batch, 1):
                    v = raw[str(i)]
                    if fmt == "dist":
                        p = np.clip(np.array([float(x) for x in v], dtype=float), 0, None)
                        if p.shape != (6,) or p.sum() <= 0: raise ValueError(f"bad dist for {i}")
                        p = p / p.sum()
                    else:
                        p = twohot(float(v))
                    out.append(Example(e.uid, e.state, {"answer": int(round(float(p @ np.arange(6))))}, None, "", {"labeler": cl.model, "soft": {"answer": p.tolist()}}))
                return out
            except Exception as ex:
                err = str(ex)[:150]
        return [Example(e.uid, e.state, None, None, "", {"label_error": err}) for e in batch]
    res = await cl.map(batches, one, desc=purpose)
    return [x for b in res for x in b]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-3.1-flash-lite")
    ap.add_argument("--guide", default="semeval_soft")
    ap.add_argument("--fmt", default="dist")
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0, help="pool: first n only (0=all)")
    ap.add_argument("--sample", type=int, default=-1)
    ap.add_argument("--maxp", type=float, default=0.3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", action="store_true")
    a = ap.parse_args()
    eb = {"provider": {"sort": "throughput", "max_price": {"prompt": a.maxp, "completion": a.maxp * 8}}}
    if a.sample >= 0: eb["seed"] = a.sample
    cl = OpenRouterClient(model=a.model, temperature=1.0 if a.sample >= 0 else 0.0, extra_body=eb, concurrency=32)
    if a.split in ("dev", "pool"):
        exs = protocol.dev(T) if a.split == "dev" else protocol.pool(T)
    else:
        exs = [Example(r["uid"], r["state"], None) for r in map(json.loads, open(D / f"{a.split}.jsonl"))]
    if a.split != "dev" and a.n: exs = exs[:a.n]
    inp = [replace(e, labels=None, gold=None) for e in exs]
    out = await label_batched(cl, inp, guide=a.guide, fmt=a.fmt, bs=a.bs, purpose=f"stsb:b:{a.tag}:{a.split}", seeds=a.seeds)
    save_jsonl(D / f"labels_{a.split}_{a.tag}.jsonl", out)
    print(a.tag, "stats", cl.stats, "failed", sum(e.labels is None for e in out))
    if a.split == "dev":
        g = np.array([e.gold["answer"] for e in exs]); s = np.array([expected(e) if e.labels else np.nan for e in out])
        ok = ~np.isnan(s); h = np.arange(len(g)) % 2 == 0
        print(f"{a.tag}: n_ok={ok.sum()} spearman all={sp(g[ok], s[ok]):.4f} A={sp(g[ok&h], s[ok&h]):.4f} B={sp(g[ok&~h], s[ok&~h]):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
