"""Select + fit calibrator on int8 OOF dev probs (5-fold CV over dev, identical recipe, held-out fold never trained on)."""
import sys, json, shutil; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.calibration import select_calibrator, calibration_metrics
T = "prompt_injections"; D = ROOT / "data/v2" / T; R = protocol.RESULTS / T
dev = protocol.dev(T); y = np.array([int(bool(e.labels["answer"])) for e in dev])
p = np.load(D / "oof_cv_gte_e8_dup3_int8.npy").astype(float)
np.save(R / "dev_probs_deployed_answer.npy", p)
cal, scores = select_calibrator(p, y)
for m, s in scores.items(): print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} acc {s['accuracy']:.3f}")
print("chosen", cal.to_dict())
raw = calibration_metrics(p, y); print("dev raw in-place", {k: round(v, 4) for k, v in raw.items() if k != "reliability"})
out = {"description": "Dev predictions = out-of-fold int8 ONNX probabilities from the 5-fold CV over dev (i4_cv.py gte_e8 dup3, int8 via i5_final.py; data/v2/prompt_injections/oof_cv_gte_e8_dup3_int8.npy): each fold model used the identical deployed recipe (pool+synth+seeds + the other 4 dev folds x3), exported to dynamic int8 ONNX exactly like the deployed model, and predicted its held-out dev fold. Re-predicted all 200 from the fold int8 files: exact match. Calibrator chosen by select_calibrator (5-fold CV NLL) on these 200 and refit on all of them.",
       "heads": {"answer": {**cal.to_dict(), "dev_cv": scores, "n_fit": int(len(y)),
                            "dev_probs": f"benchmarks/results_v2/{T}/dev_probs_deployed_answer.npy"}}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
