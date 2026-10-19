"""Batched soft labeling for toxigen: one system prompt (both questions + guidance [+ soft gold seeds]),
N numbered texts per request, reply {"1": {"toxic": p, "toxicity": [p0..p4]}, ...}. Cuts cost ~10x vs one call per text.
Usage: blabel.py dev VARIANT [VARIANT ...]   |   blabel.py pool VARIANT"""
from __future__ import annotations

import asyncio, json, sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import GUIDANCE_V1, LAB, QS, RES, TASK, dev_report, save, seeds as gold_seeds, soft_of, load  # noqa: E402
from fast_budget import FastBudget  # noqa: E402
from benchmarks.v2 import protocol as P  # noqa: E402
from ownyourjev.llm import OpenRouterClient  # noqa: E402

MODELS = {"flash": "deepseek/deepseek-v4.1-flash", "haiku": "anthropic/claude-haiku-5.5", "pro": "deepseek/deepseek-v4-pro",
          "gemlite": "google/gemini-2.5-flash-lite", "qwen": "qwen/qwen3.7-flash", "mini": "openai/gpt-5-mini"}
# variant = model_key + "_b<bs>" + ("_s" if seeds)
def parse(v):
    parts = v.split("_"); m = parts[0]; bs = 20; sd = False
    for p in parts[1:]:
        if p.startswith("b"): bs = int(p[1:])
        if p == "s": sd = True
    return m, bs, sd


def system_prompt(with_seeds: bool) -> str:
    L = ["You are a meticulous, consistent annotator of hate speech. For each numbered TEXT you answer two questions.", "",
         "### toxic", QS["toxic"].describe(), "", "### toxicity", QS["toxicity"].describe(), "",
         "## Labeling guidelines", GUIDANCE_V1.strip(), "",
         "## Output",
         'Reply with ONE JSON object mapping each text number (as a string) to {"toxic": <probability 0-1 that toxic is true>, '
         '"toxicity": [p0, p1, p2, p3, p4]} where p_i is your probability that the average annotator picks level i (sum 1). '
         'Use at most 2 decimals. Answer every number. Example: {"1": {"toxic": 0.05, "toxicity": [0.8, 0.15, 0.05, 0, 0]}, "2": {"toxic": 0.9, "toxicity": [0, 0.05, 0.25, 0.4, 0.3]}}. JSON only.']
    if with_seeds:
        L += ["", "## Examples of correct answers (from expert annotation)"]
        for s in gold_seeds(True):
            d = [s.labels["toxicity"][str(i)] for i in range(5)]
            L.append(f'TEXT: {json.dumps(s.state["text"])}\nANSWER: {json.dumps({"toxic": s.labels["toxic"], "toxicity": d})}')
    return "\n".join(L)


def _parse_item(x):
    try:
        p = float(x["toxic"]); d = np.clip(np.asarray(x["toxicity"], dtype=float), 0, None)
        if d.shape != (5,) or d.sum() <= 0 or not 0 <= p <= 1:
            return None
        return p, (d / d.sum()).tolist()
    except Exception:
        return None


async def _batch(client, sysmsg, batch, purpose):
    user = "\n".join(f"{i + 1}. {json.dumps(e.state['text'])}" for i, e in enumerate(batch))
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    try:
        raw = await client.complete_json(msgs, purpose=purpose, max_tokens=70 * len(batch) + 100)
    except Exception:
        raw = {}
    return [(_parse_item(raw.get(str(i + 1))) if isinstance(raw, dict) and isinstance(raw.get(str(i + 1)), dict) else None) for i in range(len(batch))]


async def label(examples, variant, purpose, concurrency=16):
    m, bs, sd = parse(variant)
    client = OpenRouterClient(model=MODELS[m], concurrency=concurrency, budget=FastBudget(),
                              extra_body={"provider": {"sort": "throughput", "max_price": {"prompt": 0.3, "completion": 2.0}}})
    sysmsg = system_prompt(sd)
    res = []
    for c in range(0, len(examples), 1000):
        chunk = examples[c:c + 1000]
        bl = [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        rr = await asyncio.gather(*[_batch(client, sysmsg, b, purpose) for b in bl])
        res += [x for r in rr for x in r]
        print(variant, c + len(chunk), round(client.stats["cost"], 4), flush=True)
    miss = [i for i, r in enumerate(res) if r is None]
    if miss:  # retry missing in batches of 5
        bl = [miss[i:i + 5] for i in range(0, len(miss), 5)]
        rr = await asyncio.gather(*[_batch(client, sysmsg, [examples[j] for j in b], purpose + ":retry") for b in bl])
        for b, r in zip(bl, rr):
            for j, x in zip(b, r):
                res[j] = x
    await client.aclose()
    out = [replace(e, labels=({"toxic": r[0] >= 0.5, "toxicity": int(np.argmax(r[1]))} if r else None),
                   meta={**e.meta, "labeler": MODELS[m], **({"soft": {"toxic": r[0], "toxicity": r[1]}} if r else {})}) for e, r in zip(examples, res)]
    return out, client.stats


async def run_dev(v):
    dev = P.dev(TASK)
    exs, st = await label(dev, v, f"{TASK}:dev:b:{v}")
    save(exs, LAB / f"dev_b_{v}.jsonl")
    ok = [i for i, e in enumerate(exs) if e.labels]
    if not ok:
        print(json.dumps({"variant": v, "fail": 200})); return
    p = np.array([exs[i].meta["soft"]["toxic"] for i in ok]); d = np.array([exs[i].meta["soft"]["toxicity"] for i in ok])
    r = dev_report(p, d, [dev[i] for i in ok])
    print(json.dumps({"variant": v, "fail": 200 - len(ok), "cost": round(st["cost"], 4), **{k: round(r[k], 3) for k in ["acc_es_best", "spearman_es", "auroc_es", "acc_p_best", "auroc_p"]}}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        async def _m():
            await asyncio.gather(*(run_dev(v) for v in sys.argv[2:]))
        asyncio.run(_m())
    else:
        v = sys.argv[2]
        pool = P.pool(TASK)
        exs, st = asyncio.run(label(pool, v, f"{TASK}:pool:b:{v}", concurrency=24))
        save(exs, LAB / f"pool_b_{v}.jsonl")
        print("done", v, st, "fail", sum(e.labels is None for e in exs))
