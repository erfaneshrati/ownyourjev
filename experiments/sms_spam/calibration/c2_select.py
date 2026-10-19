"""Calibration step 2: select + fit the calibrator on dev only and freeze CALIBRATION.json.

Scheme (primary): POOLED. Test probabilities are a mixture: each test text is scored by exactly one of the 5 fold
models. The calibrator is one map applied to that mixture. Stacking all 5 fold models' dev predictions (5 x 200 =
1000 rows) gives the same equal mixture of models (each fold holds ~1/5 of test) and uses every out-of-sample
prediction, instead of throwing away 4/5 of them by an arbitrary assignment. CV is grouped by dev example, so the
5 rows of one text never sit on both sides of a split. Averaging the 5 models is NOT used (test is never averaged).
Sensitivity: deterministic assignment (seeded balanced permutation, 40 dev texts per fold model), 200 rows."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
from ownyourjev.calibration import select_calibrator
T = "sms_spam"
dev = protocol.dev(T); y = np.array([int(bool(e.labels["answer"])) for e in dev])
P = np.load(ROOT / "data/v2/sms_spam/calibration_dev_probs_all_folds.npy").astype(float)  # (5, 200)
R = protocol.RESULTS / T

# primary: pooled
p_pool = P.reshape(-1); y_pool = np.tile(y, 5); g_pool = np.tile(np.arange(len(y)), 5)
cal, cv = select_calibrator(p_pool, y_pool, groups=g_pool)
# sensitivity: deterministic assignment
perm = np.random.RandomState(0).permutation(len(y)); assign = np.zeros(len(y), int); assign[perm] = np.arange(len(y)) % 5
p_asg = P[assign, np.arange(len(y))]
cal_a, cv_a = select_calibrator(p_asg, y)

np.save(R / "dev_probs_deployed.npy", P.astype(np.float32))           # (5 fold models, 200 dev) -> pooled
np.save(R / "dev_probs_deployed_assigned.npy", p_asg.astype(np.float32))
def show(name, cv):
    print(name); [print(f"  {m:12s} nll {v['nll']:.4f} ece {v['ece']:.4f} ace {v['ace']:.4f} brier {v['brier']:.4f}") for m, v in cv.items()]
show("POOLED (1000 rows, grouped 5-fold CV)", cv); print("  ->", cal.to_dict())
show("ASSIGNED (200 rows, stratified 5-fold CV)", cv_a); print("  ->", cal_a.to_dict())

out = {"description": "Fitted on dev only. Dev (200 gold, carved out of the eval set, in no fold's training data) was "
       "predicted by each of the 5 deployed int8 ONNX fold models with the same inference path as test (common.onnx_probs, "
       "length-sorted batches of 32); stored dev_probs_int8 and full-fold test re-prediction reproduce bit-exactly. "
       "Scheme: pooled, all 5 fold models' dev predictions stacked (1000 rows = 200 texts x 5 models), matching the test "
       "mixture of fold models; CV grouped by dev text. Sensitivity (deterministic 40-per-model assignment) in CALIBRATION_NOTES.md.",
       "heads": {"answer": {**cal.to_dict(), "dev_cv": cv, "n_fit": int(len(p_pool)), "n_dev_examples": int(len(y)),
                             "scheme": "pooled_5_fold_models_grouped_cv",
                             "sensitivity_assigned": {**cal_a.to_dict(), "dev_cv": cv_a, "n_fit": int(len(y))}}}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote", R / "CALIBRATION.json")
