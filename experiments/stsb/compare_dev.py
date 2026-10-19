"""Paired bootstrap on dev: candidate (probs .npy) vs (a) round-1 deployed model and (b) round-1 recipe = mean Spearman over its 6 seeds."""
import sys
from common import *
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev]); LV = np.arange(6)
cand = np.load(sys.argv[1]) @ LV
r1_dep = np.load(R / "dev_probs.npy") @ LV
r1_seeds = [np.load(D / "runs" / t / "dev_probs.npy") @ LV for t in ["F_s0", "F_s1", "F_s2", "F_s3", "R2_base_s4", "R2_base_s5"]]
def stats_(idx):
    c = sp(g[idx], cand[idx]); d = sp(g[idx], r1_dep[idx]); m = np.mean([sp(g[idx], s[idx]) for s in r1_seeds])
    return c, c - d, c - m
pt = stats_(np.arange(200))
rng = np.random.default_rng(0); B = np.array([stats_(rng.integers(0, 200, 200)) for _ in range(2000)])
print(f"cand {pt[0]:.4f} | vs r1 deployed {pt[1]:+.4f} CI [{np.percentile(B[:,1],2.5):+.4f},{np.percentile(B[:,1],97.5):+.4f}] P(>0)={np.mean(B[:,1]>0):.3f} | vs r1 recipe seed-mean {pt[2]:+.4f} CI [{np.percentile(B[:,2],2.5):+.4f},{np.percentile(B[:,2],97.5):+.4f}] P(>0)={np.mean(B[:,2]>0):.3f}")
