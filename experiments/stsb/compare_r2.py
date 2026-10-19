"""Paired bootstrap on dev: candidate vs round-2 deployed model and vs round-2 recipe member mean."""
import sys
from common import *
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev]); LV = np.arange(6)
cand = np.load(sys.argv[1]) @ LV
r2 = np.load(R / "round2" / "dev_probs.npy") @ LV
mem = [np.load(D / "runs" / t / "dev_probs.npy") @ LV for t in ["R2_allF_s0", "R2_allFi_s1", "R2_allFi_s2"]]
def f(idx):
    c = sp(g[idx], cand[idx]); return c, c - sp(g[idx], r2[idx]), c - np.mean([sp(g[idx], m[idx]) for m in mem])
pt = f(np.arange(200)); rng = np.random.default_rng(0); B = np.array([f(rng.integers(0, 200, 200)) for _ in range(2000)])
for k, n in [(1, "r2 deployed"), (2, "r2 member mean")]:
    print(f"cand {pt[0]:.4f} vs {n}: {pt[k]:+.4f} CI [{np.percentile(B[:,k],2.5):+.4f},{np.percentile(B[:,k],97.5):+.4f}] P(>0)={np.mean(B[:,k]>0):.3f}")
