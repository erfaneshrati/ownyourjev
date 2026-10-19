"""Recompute test + dev probabilities at batch size 1 (deployed serving mode) with the frozen int8 ONNX; refit temperature on b1 dev."""
import json, numpy as np
from pathlib import Path
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator
T = "banking77"; R = Path("benchmarks/results_v2/banking77")
p = OnnxPredictor(Path("data/v2/banking77/models/votesoft"), threads=4)
ev = P.eval_inputs(T); dev = P.dev(T)
pe = np.concatenate([p.probs([e.state]) for e in ev]); np.save(R / "eval_probs_b1.npy", pe)
pd = np.concatenate([p.probs([e.state]) for e in dev]); np.save(R / "dev_probs_b1.npy", pd)
p0 = np.load(R / "eval_probs.npy"); d0 = np.load(R / "dev_probs_deployed.npy")
y = np.load("experiments/banking77/calibration/dev_y.npy")
print("test: max|dp|", np.abs(pe - p0).max(), "argmax flips", int((pe.argmax(1) != p0.argmax(1)).sum()))
print("dev: max|dp|", np.abs(pd - d0).max(), "argmax flips", int((pd.argmax(1) != d0.argmax(1)).sum()), "dev acc b1", (pd.argmax(1) == y).mean())
cal, sc = select_calibrator(pd, y, methods=("none", "temperature"))
for m, s in sc.items(): print(m, {k: round(s[k], 4) for k in ("nll", "ece", "ace", "brier", "mean_conf", "accuracy")})
print("chosen", cal.to_dict())
json.dump({"description": "Frozen votesoft int8 ONNX (OnnxPredictor threads=4), one text per call, same preprocessing; frozen eval_probs.npy came from batches of 64 (predict_eval.py). Temperature refit on batch-1 dev predictions (dev not in training texts), 5-fold CV NLL over none/temperature.",
           "heads": {"answer": {"probs": str(R / "eval_probs_b1.npy"), "dev_probs": str(R / "dev_probs_b1.npy")}},
           "calibration": {"answer": cal.to_dict()}, "dev_cv_b1": {"answer": sc}}, open(R / "BATCH1.json", "w"), indent=1, default=float)
