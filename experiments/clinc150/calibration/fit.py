"""Fit the calibrator on dev only. 151 classes / 200 dev examples: most classes have 0-3 dev examples, so
per-class methods (vector scaling: 151 biases; isotonic: 151 one-vs-rest regressors) cannot be estimated.
We report their CV for transparency but select only among none / temperature (1 parameter)."""
import json
from collections import Counter
import numpy as np
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *
from ownyourjev.calibration import cv_scores, select_calibrator

p = np.load(P.RESULTS / TASK / "dev_probs_deployed.npy").astype(float)
y = np.array([OPTS.index(e.labels["answer"]) for e in P.dev(TASK)])
c = Counter(y); print("classes in dev", len(c), "max/class", max(c.values()), "classes with >=5:", sum(v >= 5 for v in c.values()))
allcv = cv_scores(p, y, ("none", "temperature", "vector", "isotonic"))
for s in (1, 2):  # seed stability of the none vs temperature comparison
    print("seed", s, {m: round(v["nll"], 4) for m, v in cv_scores(p, y, ("none", "temperature"), seed=s).items()})
cal, sel = select_calibrator(p, y, methods=("none", "temperature"))
for m, v in allcv.items():
    print(f"{m:12s} nll {v['nll']:.4f} ece {v['ece']:.4f} ace {v['ace']:.4f} brier {v['brier']:.4f} acc {v['accuracy']:.3f} conf {v['mean_conf']:.3f}")
print("chosen", cal.to_dict())
desc = ("Dev predictions: the exact deployed artifact (data/v2/clinc150/models/E_lr5_q_only_Wi/model.int8.onnx, mlp.Wi-only int8) "
        "run through the predict_eval.py path (OnnxPredictor, batch 64, then apply_threshold tau=0.4) on the 200 gold dev examples; "
        "the deployed model never trained on dev, so these are out-of-sample. Verified: re-predicting the first 256 eval_inputs "
        "reproduces the frozen eval_probs.npy with max abs diff 0.0. Calibrated vector = the deployed (post-oos-rule) probability vector. "
        "Candidates restricted to none/temperature on dev (151 classes, 200 examples: per-class vector/isotonic not estimable); "
        "5-fold CV NLL picks the method, refit on all 200 dev.")
json.dump({"description": desc, "heads": {"answer": {**cal.to_dict(), "dev_cv": sel, "dev_cv_all_methods_for_reference": allcv,
          "candidates": ["none", "temperature"], "n_fit": int(len(y))}}},
          open(P.RESULTS / TASK / "CALIBRATION.json", "w"), indent=1)
