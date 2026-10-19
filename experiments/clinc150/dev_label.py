"""Compare labeling variants on dev (200 gold). Usage: dev_label.py <variant> [model] [bs]"""
import asyncio, sys, json
from collections import Counter
from benchmarks.v2 import protocol as P
from experiments.clinc150 import blabel

variant = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash"
bs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
dev = P.dev("clinc150"); seeds = P.seeds("clinc150")
kw = dict(model=model, bs=bs, purpose=f"clinc-dev-{variant}")
if variant == "nodesc": kw.update(with_desc=False)
if variant == "noseeds": pass
else: kw.update(seeds=seeds)
out = asyncio.run(blabel.label(dev, **kw))
pred = [(o.labels or {}).get("answer") for o in out]
gold = [d.labels["answer"] for d in dev]
acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
h1 = sum(p == g for p, g in zip(pred[:100], gold[:100])) / 100
h2 = sum(p == g for p, g in zip(pred[100:], gold[100:])) / 100
print(f"{variant} {model} bs={bs}: acc={acc:.3f} (A {h1:.2f} / B {h2:.2f}) pred_oos={pred.count('oos')} gold_oos={gold.count('oos')}")
for p, g, d in zip(pred, gold, dev):
    if p != g: print(f"  gold={g:24s} pred={p:24s} | {d.state['utterance']}")
