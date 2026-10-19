"""Dev-only calibrator selection (5-fold CV, NLL) on the deployed model's dev predictions; also seeds 1-4 for stability."""
import json, numpy as np
from ownyourjev.calibration import select_calibrator, cv_scores, calibration_metrics
p = np.load("benchmarks/results_v2/banking77/dev_probs_deployed.npy"); y = np.load("experiments/banking77/calibration/dev_y.npy")
cal, sc = select_calibrator(p, y)  # default: none / temperature / vector / isotonic
for m, s in sc.items(): print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} conf {s['mean_conf']:.3f} acc {s['accuracy']:.3f}")
print("chosen:", cal.method, {k: v for k, v in cal.params.items() if k != 'bias'})
stab = {}
for seed in range(1, 5):
    s = cv_scores(p, y, seed=seed); stab[seed] = {m: s[m]["nll"] for m in s}
    print("seed", seed, {m: round(v, 4) for m, v in stab[seed].items()}, "best", min(stab[seed], key=stab[seed].get))
raw = calibration_metrics(p, y); print("raw in-sample", {k: round(raw[k], 4) for k in ("nll", "ece", "brier", "mean_conf", "accuracy")})
json.dump({"chosen": cal.to_dict(), "dev_cv": sc, "seed_stability_nll": stab}, open("experiments/banking77/calibration/select.json", "w"), indent=1, default=float)
