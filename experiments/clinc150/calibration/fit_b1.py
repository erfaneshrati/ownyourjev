"""Refit the calibrator on batch-1 dev predictions (none/temperature only, 5-fold CV NLL) and write BATCH1.json."""
import json
import numpy as np
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *
from ownyourjev.calibration import select_calibrator, cv_scores

p = np.load("data/v2/clinc150/dev_probs_b1.npy").astype(float)
y = np.array([OPTS.index(e.labels["answer"]) for e in P.dev(TASK)])
cal, sel = select_calibrator(p, y, methods=("none", "temperature"))
for s in (1, 2):
    print("seed", s, {m: round(v["nll"], 4) for m, v in cv_scores(p, y, ("none", "temperature"), seed=s).items()})
for m, v in sel.items():
    print(f"{m:12s} nll {v['nll']:.4f} ece {v['ece']:.4f} ace {v['ace']:.4f} brier {v['brier']:.4f} acc {v['accuracy']:.3f}")
print("chosen", cal.to_dict())
desc = ("Batch-size-1 predictions of the frozen deployed artifact (E_lr5_q_only_Wi/model.int8.onnx, OnnxPredictor threads=8, "
        "one text per call) followed by the deployed oos rule apply_threshold(tau=0.4), baked into the vector exactly as in the "
        "frozen eval_probs.npy (so argmax flips == deployed decision flips). Dev predictions computed identically; calibrator refit "
        "on batch-1 dev with none/temperature, 5-fold CV NLL.")
json.dump({"description": desc,
           "heads": {"answer": {"probs": "data/v2/clinc150/eval_probs_b1.npy", "dev_probs": "data/v2/clinc150/dev_probs_b1.npy"}},
           "calibration": {"answer": {**cal.to_dict(), "dev_cv": sel, "n_fit": int(len(y))}}},
          open(P.RESULTS / TASK / "BATCH1.json", "w"), indent=1)
