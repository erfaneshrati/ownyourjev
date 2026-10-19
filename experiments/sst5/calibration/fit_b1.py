"""Refit calibrator on batch-1 dev predictions (argmax-preserving methods only: none/temperature); write BATCH1.json."""
import json
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.calibration import select_calibrator
from ownyourjev.export import OnnxPredictor
R = "benchmarks/results_v2/sst5"
dev = P.dev("sst5")
p = np.load(f"{R}/dev_probs_b1.npy"); pb = np.load(f"{R}/dev_probs_deployed.npy")
y = np.array([int(min(max(round(float(e.gold["answer"])), 0), 4)) for e in dev])
# determinism check: batch-1 is a function of the text alone
pr = OnnxPredictor("data/v2/sst5/models/final", threads=8)
rep = np.concatenate([pr.probs([e.state]) for e in dev[:20]]); print("b1 repeat max diff", np.abs(rep - p[:20]).max())
print("dev acc b1", (p.argmax(1) == y).mean(), "batched", (pb.argmax(1) == y).mean(), "dev flips", int((p.argmax(1) != pb.argmax(1)).sum()))
cal, scores = select_calibrator(p, y, methods=("none", "temperature"))
for m, s in scores.items(): print(m, {k: round(v, 4) for k, v in s.items()})
print("chosen", cal.to_dict())
json.dump({"description": "sst5/answer at batch size 1 (serving mode): deployed partial-int8 ONNX (data/v2/sst5/models/final/model.int8.onnx), "
           "OnnxPredictor threads=8, one text per call (experiments/sst5/calibration/predict_b1.py). Calibrator refit on batch-1 dev "
           "probs (model never trained on dev), 5-fold CV NLL over none/temperature.",
           "heads": {"answer": {"probs": f"{R}/eval_probs_b1.npy", "dev_probs": f"{R}/dev_probs_b1.npy"}},
           "calibration": {"answer": {**cal.to_dict(), "dev_cv": scores, "n_fit": int(len(y))}}},
          open(f"{R}/BATCH1.json", "w"), indent=1, default=float)
