"""Fit + select the calibrator on dev only (5-fold CV over none/temperature/vector/isotonic), write CALIBRATION.json."""
import json
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.calibration import select_calibrator, cv_scores
p = np.load("benchmarks/results_v2/sst5/dev_probs_deployed.npy")
y = np.array([int(min(max(round(float(e.gold["answer"])), 0), 4)) for e in P.dev("sst5")])
print("dev class counts", np.bincount(y, minlength=5))
cal, scores = select_calibrator(p, y)
for m, s in scores.items(): print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} acc {s['accuracy']:.3f} conf {s['mean_conf']:.3f}")
print("chosen", cal.to_dict())
# seed-stability check of the CV choice (dev only)
for sd in (1, 2, 3, 4):
    sc = cv_scores(p, y, seed=sd); print("seed", sd, {m: round(v["nll"], 4) for m, v in sc.items()})
json.dump({"description": "sst5/answer: dev probs = deployed partial-int8 ONNX (data/v2/sst5/models/final/model.int8.onnx) "
           "run with the predict_final.py path (OnnxPredictor, contiguous batches of 32) on the 200 gold dev examples; the "
           "model never trained on dev (ub_s1, gold_fold=None), so these are out-of-sample. Reproduces frozen dev_probs.npy "
           "exactly and frozen eval_probs.npy exactly (diff 0.0) on 4 eval batches with the same batching. Method chosen by "
           "ownyourjev.calibration.select_calibrator (5-fold stratified CV NLL, seed 0) over none/temperature/vector/isotonic, "
           "refit on all 200 dev examples. Gold = score rounded to nearest of 5 levels.",
           "heads": {"answer": {**cal.to_dict(), "dev_cv": scores, "n_fit": int(len(y))}}},
          open("benchmarks/results_v2/sst5/CALIBRATION.json", "w"), indent=1, default=float)
