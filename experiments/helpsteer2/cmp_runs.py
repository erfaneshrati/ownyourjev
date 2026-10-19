"""Compare two groups of runs on gold dev: per-seed means, seed-ensemble, and paired bootstrap (resampling dev examples) of
the seed-averaged per-run metric. Usage: cmp_runs.py "A1 A2 ..." "B1 B2 ..." [--cv k] (with --cv, names are prefixes of fold runs)"""
import sys
import numpy as np
from common import *
from postproc import blend_hc
M = ROOT / "data/v2/helpsteer2/models"
dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
cv = int(sys.argv[sys.argv.index("--cv") + 1]) if "--cv" in sys.argv else 0
def load(n):
    if cv:
        P = np.zeros((200, 5, 5))
        for i in range(cv):
            idx = [j for j in range(200) if j % cv == i]; P[idx] = np.load(M / f"{n}_f{i}" / "dev_probs10.npy")[:, :5]
    else:
        P = np.load(M / n / "dev_probs10.npy")[:, :5]
    return blend_hc(P) @ np.arange(5)
def met(E, ix):
    return np.array([sp(E[ix, j], G[ix, j]) for j in range(5)])
grp = [sys.argv[1].split(), sys.argv[2].split()]
Es = [[load(n) for n in g] for g in grp]
full = np.arange(200)
for g, E in zip(grp, Es):
    per = np.array([met(e, full) for e in E]); ens = met(np.mean([(e - e.mean(0)) / e.std(0) for e in E], 0), full)
    print(f"{' '.join(g)[:40]:40s} per-seed mean " + " ".join(f"{x:.3f}" for x in per.mean(0)) + f" | {per.mean():.3f}   seed-ens {ens.mean():.3f} (h {ens[0]:.3f} c {ens[1]:.3f})")
rng = np.random.default_rng(0); d_all, d_hc = [], []
for _ in range(1000):
    ix = rng.integers(0, 200, 200)
    a = np.mean([met(e, ix) for e in Es[0]], 0); b = np.mean([met(e, ix) for e in Es[1]], 0)
    d_all.append(a.mean() - b.mean()); d_hc.append(a[:2].mean() - b[:2].mean())
for nm, d in [("mean5", d_all), ("h/c", d_hc)]:
    d = np.array(d); print(f"A-B {nm}: {d.mean():+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  P(>0) {np.mean(d > 0):.2f}")
