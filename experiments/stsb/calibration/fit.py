"""Fit the calibrator on the 200 dev pairs only (out-of-sample deployed predictions), write CALIBRATION.json.
Also reports the effect on the dev expected-score Spearman (5-fold CV out-of-fold + refit-on-all)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
from ownyourjev.calibration import Calibrator, select_calibrator, cv_scores
from sklearn.model_selection import StratifiedKFold, KFold

p = np.load(R / "dev_probs_deployed.npy")
dev = protocol.dev(T); g = np.array([float(e.gold["answer"]) for e in dev])
y = np.array([int(min(max(round(x), 0), 5)) for x in g])
print("level counts", np.bincount(y, minlength=6))
cal, scores = select_calibrator(p, y)
print(f"{'method':12s} {'nll':>7s} {'ece':>7s} {'ace':>7s} {'brier':>7s} {'acc':>6s} {'conf':>6s}")
for m, s in scores.items():
    print(f"{m:12s} {s['nll']:7.4f} {s['ece']:7.4f} {s['ace']:7.4f} {s['brier']:7.4f} {s['accuracy']:6.3f} {s['mean_conf']:6.3f}")
print("chosen", cal.method, cal.params)
LV = np.arange(6)
# expected-score Spearman: raw vs calibrated (refit on all dev, and out-of-fold for each method)
print("dev spearman raw", round(sp(g, p @ LV), 4), "| calibrated (refit-all)", round(sp(g, cal.apply(p) @ LV), 4))
spl = list(StratifiedKFold(5, shuffle=True, random_state=0).split(p[:, 0], y)) if np.bincount(y).min() >= 5 else list(KFold(5, shuffle=True, random_state=0).split(p))
sp_oof = {}
for m in scores:
    oof = np.zeros_like(p)
    for tr, te in spl:
        oof[te] = Calibrator(m).fit(p[tr], y[tr]).apply(p[te])
    sp_oof[m] = round(sp(g, oof @ LV), 4)
print("dev spearman, out-of-fold per method", sp_oof)
out = {"description": ("STS-B round-2 deployed model (data/v2/stsb/models/final_r2/model.int8.onnx, OnnxPredictor threads=4, batch 1, "
                       "same path as experiments/stsb/predict_eval.py). The model never trained on dev or seeds (train2.py excludes their uids; "
                       "re-paired/generated pairs exclude dev/seed sentences; soup members R2_allF_s0/R2_allFi_s1/R2_allFi_s2 have add_gold empty), "
                       "so dev predictions are out-of-sample. Gold level = dev gold score rounded to the nearest of 0..5. "
                       "Method chosen by 5-fold CV NLL on the 200 dev pairs (select_calibrator), refit on all 200."),
       "heads": {"answer": {**cal.to_dict(), "dev_cv": scores, "n_fit": int(len(y)),
                            "dev_expected_score_spearman": {"raw": round(sp(g, p @ LV), 4), "calibrated_refit_all": round(sp(g, cal.apply(p) @ LV), 4), "oof_by_method": sp_oof}}}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote", R / "CALIBRATION.json")
