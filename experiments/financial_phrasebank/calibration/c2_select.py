"""Select + fit the calibrator on the 200 gold dev examples (out-of-sample int8 dev probs from c1), write CALIBRATION.json."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ag_news"))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.calibration import select_calibrator, calibration_metrics
T = "financial_phrasebank"; R = ROOT / "benchmarks/results_v2" / T
q = protocol.task_spec(T).parsed["answer"]
dev = protocol.dev(T); y = np.array([q.options.index(e.labels["answer"]) for e in dev])
p = np.load(R / "dev_probs_deployed.npy")
assert np.array_equal(p, np.load(ROOT / "experiments/financial_phrasebank/calibration/dev_probs_c1.npy"))
print("class counts", np.bincount(y, minlength=3), "options", q.options)
cv = cv_scores_all = __import__("ownyourjev.calibration", fromlist=["cv_scores"]).cv_scores(p, y)  # full table, reported
# Restrict to argmax-preserving methods: vector/isotonic change the top label (vector: 11/200 dev argmax flips), which would
# make the calibrated probabilities contradict the frozen deployed decisions; vector's CV NLL gain (0.004) is within fold noise.
cal, _ = select_calibrator(p, y, methods=("none", "temperature"))
for m, s in cv.items():
    print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} conf {s['mean_conf']:.3f} acc {s['accuracy']:.3f}")
# seed robustness of the choice (different fold splits), for the notes only
for sd in (1, 2, 3):
    from ownyourjev.calibration import cv_scores
    s = cv_scores(p, y, seed=sd); print("seed", sd, {m: round(v["nll"], 4) for m, v in s.items()})
print("chosen", cal.method, cal.params)
pc = cal.apply(p); assert (pc.argmax(1) == p.argmax(1)).all() or cal.method in ("vector", "isotonic")
print("dev argmax changes after calibration:", int((pc.argmax(1) != p.argmax(1)).sum()))
d = cal.to_dict(); d["params"] = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else v) for k, v in d["params"].items()}
d["dev_cv"] = cv; d["n_fit"] = len(y)
out = {"description": "Calibrator fitted on the 200 gold dev examples using the deployed int8 ONNX model's own dev predictions "
       "(model.int8.onnx via OnnxPredictor, identical to finalize_probs.py; re-predicted, max |diff| 0.0 vs saved dev probs and "
       "vs the first 256 frozen test probs). Dev uids were dropped from training, no dev text in the train pool, so dev "
       "predictions are out-of-sample. dev_cv covers none/temperature/vector/isotonic (5-fold stratified CV); selection restricted to argmax-preserving methods (none, temperature) because vector flips 11/200 dev top labels (calibrated probs would contradict the frozen decisions) for a CV NLL gain of 0.004 (within fold-seed noise) and isotonic is far worse; lowest CV NLL among those = none.",
       "heads": {"answer": d}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(d["params"]))
