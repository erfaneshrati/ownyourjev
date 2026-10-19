"""Paired bootstrap over dev articles: candidate (ensemble of int8 fold-model prediction sets, averaged) vs baseline.
usage: paired_boot2.py <out_name> <baseline_prefixes comma> <candidate_prefixes comma>   (prefix X -> dev_hold/X{k}_int8_dev.npy)"""
import json, sys
import numpy as np
from scipy import stats
from common import *
dev = P.dev(T); groups = sorted({e.group for e in dev})
def per_article(prefixes):
    out = []
    for k in range(5):
        p = np.mean([np.load(DATA / "dev_hold" / f"{px}{k}_int8_dev.npy") for px in prefixes], 0); s = p @ np.arange(5)
        A = np.full((len(groups), 4), np.nan)
        for a, g in enumerate(groups):
            idx = [i for i, e in enumerate(dev) if e.group == g]
            for j, d in enumerate(DIMS):
                y = [dev[i].labels[d] for i in idx]
                if np.std(y) > 0 and np.std(s[idx, j]) > 0: A[a, j] = stats.spearmanr(y, s[idx, j]).statistic
        out.append(A)
    return np.array(out)
b, c = per_article(sys.argv[2].split(",")), per_article(sys.argv[3].split(","))
sc = lambda X, idx: float(np.mean([np.nanmean(X[:, idx, j]) for j in range(4)]))
al = np.arange(len(groups)); rng = np.random.default_rng(0)
diffs = np.array([sc(c, i) - sc(b, i) for i in (rng.integers(0, len(groups), len(groups)) for _ in range(5000))])
res = {"baseline": sys.argv[2], "candidate": sys.argv[3], "baseline_dev": sc(b, al), "candidate_dev": sc(c, al), "diff": sc(c, al) - sc(b, al),
       "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "p_diff_le_0": float((diffs <= 0).mean()), "n_articles": len(groups),
       "per_dim_candidate": {d: float(np.nanmean(c[:, :, j])) for j, d in enumerate(DIMS)}}
(RES / f"{sys.argv[1]}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
