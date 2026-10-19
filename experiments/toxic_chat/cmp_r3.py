"""Dev comparison of a candidate (int8 dev probs) vs the round-2 frozen model: threshold rule, CV, AP, paired bootstrap."""
import sys, numpy as np
from sklearn.metrics import average_precision_score
from common import *
from threshold import smooth_threshold, cv_boot_threshold_f1
dev = protocol.dev(T)
ref = np.load(D / "runs/soupAB_sq0.75/dev_probs_int8_full.npy")
for cand in sys.argv[1:]:
    p = np.load(D / "runs" / cand)
    for i, h in enumerate(HEADS):
        y = np.array([e.gold[h] for e in dev])
        t1, _ = smooth_threshold(y, p[:, i]); t0, _ = smooth_threshold(y, ref[:, i])
        rng = np.random.default_rng(0); dF, dA = [], []
        for _ in range(2000):
            ii = rng.integers(0, 200, 200)
            if y[ii].sum() == 0: continue
            dF.append(f1_at(y[ii], p[ii, i], t1) - f1_at(y[ii], ref[ii, i], t0))
            dA.append(average_precision_score(y[ii], p[ii, i]) - average_precision_score(y[ii], ref[ii, i]))
        print(f"{cand} {h}: t={t1} F1@t={f1_at(y,p[:,i],t1):.3f} (ref {f1_at(y,ref[:,i],t0):.3f}@{t0}) cv_rule={cv_boot_threshold_f1(y,p[:,i],reps=10):.3f} (ref {cv_boot_threshold_f1(y,ref[:,i],reps=10):.3f}) "
              f"AP={average_precision_score(y,p[:,i]):.3f} (ref {average_precision_score(y,ref[:,i]):.3f}) dF1={np.mean(dF):.3f} [{np.percentile(dF,2.5):.3f},{np.percentile(dF,97.5):.3f}] dAP={np.mean(dA):.3f} [{np.percentile(dA,2.5):.3f},{np.percentile(dA,97.5):.3f}]")
