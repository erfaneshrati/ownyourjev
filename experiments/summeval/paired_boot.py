"""Paired bootstrap over dev articles: round-2 vs round-1 deployed int8 fold models (dev score = mean over the 5 fold models
of the per-article Spearman, averaged over articles and the 4 dims)."""
import json
import numpy as np
from scipy import stats
from common import *
dev = P.dev(T); groups = sorted({e.group for e in dev})
def per_article(prefix):
    # (n_folds, n_articles, 4) per-article Spearman
    out = []
    for k in range(5):
        p = np.load(DATA / "dev_hold" / f"{prefix}{k}_int8_dev.npy")
        s = p @ np.arange(5)
        A = np.full((len(groups), 4), np.nan)
        for a, g in enumerate(groups):
            idx = [i for i, e in enumerate(dev) if e.group == g]
            for j, d in enumerate(DIMS):
                y = [dev[i].labels[d] for i in idx]
                if np.std(y) > 0 and np.std(s[idx, j]) > 0:
                    A[a, j] = stats.spearmanr(y, s[idx, j]).statistic
        out.append(A)
    return np.array(out)
r1, r2 = per_article("F"), per_article("G")
def score(X, idx):  # mean over folds, dims (mean over articles per dim, like group_spearman)
    return float(np.mean([np.nanmean(X[:, idx, j]) for j in range(4)]))
all_idx = np.arange(len(groups))
d0 = score(r2, all_idx) - score(r1, all_idx)
rng = np.random.default_rng(0); diffs = []
for _ in range(5000):
    idx = rng.integers(0, len(groups), len(groups)); diffs.append(score(r2, idx) - score(r1, idx))
diffs = np.array(diffs)
res = {"round1_dev": score(r1, all_idx), "round2_dev": score(r2, all_idx), "diff": d0, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
       "p_diff_le_0": float((diffs <= 0).mean()), "n_articles": len(groups),
       "per_dim_round1": {d: float(np.nanmean(r1[:, :, j])) for j, d in enumerate(DIMS)}, "per_dim_round2": {d: float(np.nanmean(r2[:, :, j])) for j, d in enumerate(DIMS)}}
(RES / "round2_dev_paired_bootstrap.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
