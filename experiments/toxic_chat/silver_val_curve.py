"""Threshold curve of a model vs. the labeler-ensemble labels on the held-out silver val split (never trained on)."""
import sys, random, numpy as np, torch
from common import *
from mh import onnx_probs, texts_of
name = sys.argv[1]; files = sys.argv[2:]
block = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
pool_uids = {e.uid for e in protocol.pool(T)}
by = {}
for f in files:
    for e in load_jsonl(f):
        if e.uid in block or e.uid not in pool_uids or e.labels is None:
            continue
        by.setdefault(e.uid, [e, []])[1].append([soft(e, h) for h in HEADS])
items = [(e, np.mean(ys, 0)) for (e, ys) in by.values()]
rng = random.Random(0); rng.shuffle(items)
val = items[:max(200, len(items) // 10)]
p = onnx_probs(D / "models" / name, texts_of([e for e, _ in val]), "model.int8.onnx", threads=8)
for i, h in enumerate(HEADS):
    y = np.array([s[i] >= 0.5 for _, s in val])
    print(h, "n_pos", y.sum(), "of", len(y), {t: round(f1_at(y, p[:, i], t), 3) for t in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]})
