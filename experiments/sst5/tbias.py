import sys; sys.path.insert(0,"experiments/sst5")
from lab import *; from calib import fit_bias, apply
p = probs_of(load("dev_v1_deepseek-v4.1-flash"), [e.uid for e in DEV]); y = DEVY
perm = np.random.default_rng(123).permutation(200) % 2
for k in (0, 1):
    b = fit_bias(p[perm != k], y[perm != k]); print("fold", k, "bias", b.round(2), "held acc raw", (p[perm==k].argmax(1)==y[perm==k]).mean(), "->", (apply(p[perm==k], b)==y[perm==k]).mean())
b = fit_bias(p, y); print("full", b.round(2))
