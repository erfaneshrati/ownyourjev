"""Labeling harness for sst5 (dev-side experiments + pool labeling)."""
import asyncio, json, sys, os
from pathlib import Path
import numpy as np
from scipy import stats
from benchmarks.v2 import protocol as P
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
from ownyourjev.data import Example

T = "sst5"
Q = P.task_spec(T).parsed
OUT = Path("data/v2/sst5/labels"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = P.seeds(T); DEV = P.dev(T)
DEVY = np.array([e.labels["answer"] for e in DEV])

def client(model, temperature=0.0, seed=None, pricey=False):
    eb = {}
    if seed is not None: eb["seed"] = seed
    if pricey: eb["provider"] = {"sort": "throughput"}
    return OpenRouterClient(model=model, temperature=temperature, extra_body=eb or None)

def save(name, exs):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for e in exs:
            f.write(json.dumps({"uid": e.uid, "labels": e.labels, "soft": (e.meta.get("soft") or {}).get("answer")}) + "\n")

def load(name):
    return {r["uid"]: r for r in map(json.loads, open(OUT / f"{name}.jsonl"))}

async def run(name, model, exs, *, guidance=None, seeds=True, distribution=True, temperature=0.0, seed=None, pricey=False, questions=None, preamble=None):
    c = client(model, temperature, seed, pricey)
    res = await label_examples(c, questions or Q, exs, purpose=f"sst5:{name}", seeds=SEEDS if seeds else None,
                               guidance=guidance, distribution=distribution, system_preamble=preamble, progress=False)
    await c.aclose()
    save(name, res)
    print(name, "cost", round(c.stats["cost"], 4), "calls", c.stats["calls"], "cached", c.stats["cached"], file=sys.stderr)
    return res

def probs_of(lab, uids):
    out = []
    for u in uids:
        r = lab.get(u)
        if r is None or r["labels"] is None:
            out.append(np.full(5, .2)); continue
        if r.get("soft"): out.append(np.array(r["soft"], float))
        else:
            v = np.zeros(5); v[r["labels"]["answer"]] = 1; out.append(v)
    return np.array(out)

def report(name, p=None, y=DEVY, verbose=True):
    if p is None: p = probs_of(load(name), [e.uid for e in DEV])
    am = p.argmax(1); es = p @ np.arange(5)
    acc = (am == y).mean(); sp = stats.spearmanr(es, y).statistic
    h = len(y) // 2
    accA, accB = (am[:h] == y[:h]).mean(), (am[h:] == y[h:]).mean()
    cm = np.zeros((5, 5), int)
    for a, b in zip(y, am): cm[a, b] += 1
    if verbose:
        print(f"{name}: acc={acc:.3f} (A {accA:.3f} B {accB:.3f}) spearman={sp:.3f} mae={np.abs(am-y).mean():.3f} pred_dist={np.bincount(am,minlength=5)}")
        print(cm)
    return acc
