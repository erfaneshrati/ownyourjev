import json, numpy as np
from ownyourjev.calibration import select_calibrator
p = np.load("benchmarks/results_v2/banking77/dev_probs_deployed.npy"); y = np.load("experiments/banking77/calibration/dev_y.npy")
cal, sc = select_calibrator(p, y, methods=("none", "temperature"))
full = json.load(open("experiments/banking77/calibration/select.json"))
assert cal.method == "temperature"
assert (cal.apply(p).argmax(1) == p.argmax(1)).all()
out = {"description": ("Deployed votesoft int8 ONNX (model.int8.onnx, OnnxPredictor threads=4, batches of 64, as predict_eval.py) on the 200 dev "
        "examples; dev texts are not training texts (train = pool + 20 seeds; 0 uid and 0 normalised-text overlap), so predictions are "
        "out-of-sample, though dev chose the epoch and was in the labeling prompt. Re-prediction of 128 eval inputs matches frozen test probs "
        "exactly (max diff 0.0). 5-fold CV (NLL) on dev. Candidates restricted to none/temperature because only these keep the top-1 label "
        "(the brief requires decisions to stay unchanged): vector scaling changed the argmax (OOF dev acc 0.890 -> 0.865), had worse ECE/Brier "
        "than temperature, and its NLL edge was unstable across CV seeds; isotonic OvR overfit badly (NLL 1.53). Full 4-method table in dev_cv_all_methods."),
       "heads": {"answer": {**cal.to_dict(), "dev_cv": sc, "dev_cv_all_methods": full["dev_cv"],
                            "dev_cv_seed_stability_nll": full["seed_stability_nll"], "n_fit": int(len(y))}}}
json.dump(out, open("benchmarks/results_v2/banking77/CALIBRATION.json", "w"), indent=1, default=float)
print(cal.params, {m: round(s["nll"], 4) for m, s in sc.items()})
