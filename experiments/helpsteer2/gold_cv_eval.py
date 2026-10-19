"""Assemble 5-fold gold-CV dev predictions (each fold predicted by a model trained on the other 160 dev + 20 seeds + LLM labels)
and compare with a no-gold model on the same 200 dev examples; paired bootstrap of the mean-Spearman difference."""
import sys
import numpy as np
from common import *
from postproc import blend_hc
M = ROOT / "data/v2/helpsteer2/models"
prefix, k, base = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
P = np.zeros((200, 5, 5))
for i in range(k):
    idx = [j for j in range(200) if j % k == i]; P[idx] = np.load(M / f"{prefix}_f{i}" / "dev_probs10.npy")[:, :5]
def ms(Q, ix=slice(None)):
    E = Q @ np.arange(5); return [sp(E[ix, j], G[ix, j]) for j in range(5)]
def show(nm, Q):
    for tag, R in [("raw", Q), ("blend", blend_hc(Q))]:
        rs = ms(R); print(f"{nm:12s} {tag:5s} " + " ".join(f"{r:.3f}" for r in rs) + f" mean {np.mean(rs):.3f}")
show(prefix, P)
rng = np.random.default_rng(0)
for b in base:
    B = np.load(M / b / "dev_probs10.npy")[:, :5]; show(b, B)
    diffs = []
    for _ in range(1000):
        ix = rng.integers(0, 200, 200)
        diffs.append(np.mean(ms(blend_hc(P), ix)) - np.mean(ms(blend_hc(B), ix)))
    diffs = np.array(diffs); print(f"  paired bootstrap (blend) {prefix} - {b}: mean {diffs.mean():.3f}, 95% CI [{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}], P(>0) {np.mean(diffs > 0):.2f}")
