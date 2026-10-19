"""Fit the post-hoc calibrator on the 200 dev predictions of the deployed model (select_calibrator, 5-fold CV
over none/temperature/vector/isotonic, lowest NLL) and freeze CALIBRATION.json. No test labels are read."""
import json
import numpy as np
from ownyourjev.calibration import select_calibrator, calibration_metrics
from benchmarks.v2 import protocol as P
from benchmarks.v2.ledger import log_experiment

T = "emotion"
q = P.task_spec(T).parsed["answer"]
dev = P.dev(T)
y = np.array([q.options.index(e.labels["answer"]) for e in dev])
p = np.load(P.RESULTS / T / "dev_probs_deployed.npy").astype(float)
print("class counts", dict(zip(q.options, np.bincount(y, minlength=len(q.options)).tolist())))
cv_all = select_calibrator(p, y)[1]  # full 4-method table, kept for the record
for m, s in cv_all.items():
    print(f"[all] {m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} brier {s['brier']:.4f} acc {s['accuracy']:.3f}")
# Restricted to argmax-preserving methods so the calibrated top label always equals the deployed decision
# (brief: decisions must not change). Vector scaling moved 4% of dev argmaxes and fits a 'surprise' bias from 2 dev examples.
cal, cv = select_calibrator(p, y, methods=("none", "temperature"))
for m, s in cv.items():
    print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} conf {s['mean_conf']:.3f} acc {s['accuracy']:.3f}")
print("chosen", cal.to_dict())
fitted = calibration_metrics(cal.apply(p), y)
assert (cal.apply(p).argmax(1) == p.argmax(1)).all()
print("argmax preserved on dev:", float((cal.apply(p).argmax(1) == p.argmax(1)).mean()))
out = {"description": "Dev predictions: benchmarks/results_v2/emotion/dev_probs_deployed.npy, produced by the exact deployed "
       "artifact (fp32 ONNX data/v2/emotion/models/g1_10k_fp32/model.onnx via OnnxPredictor, threads 8, batch 64) in the same "
       "run that wrote the frozen eval_probs.npy; re-prediction of 128 dev + 128 test inputs matched exactly (max abs diff 0). "
       "The model trained only on LLM-labeled pool texts, never on dev texts (dev half A informed the labeling guidance only), "
       "so dev predictions are out-of-sample. Method chosen by select_calibrator (5-fold stratified CV on the 200 dev, lowest NLL) "
       "restricted to none/temperature, refit on all 200. Restriction justified on dev: only these keep the top-1 label equal to "
       "the deployed decision; vector scaling had lower CV NLL (0.561 vs 0.615) but moved 4% of dev argmaxes and fits a per-class bias "
       "for 'surprise' from 2 dev examples; isotonic OvR was worst on NLL (0.890). Full 4-method table in dev_cv_all_methods.",
       "heads": {"answer": {**cal.to_dict(), "dev_cv": cv, "dev_cv_all_methods": cv_all, "n_fit": int(len(y))}}}
(P.RESULTS / T / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
summ = {m: {"nll": round(s["nll"], 4), "ece": round(s["ece"], 4), "brier": round(s["brier"], 4)} for m, s in cv.items()}
log_experiment(task=T, round=99, kind="selection", idea="post-hoc calibration", setting="calibration",
               dev={"answer": summ, "all_methods": {m: round(v["nll"], 4) for m, v in cv_all.items()}, "chosen": cal.method,
                    "note": "supersedes the earlier round-99 line that picked vector before restricting to argmax-preserving methods"},
               change=f"answer: {cal.method} {json.dumps(cal.params, default=float)[:200]}",
               outcome="success" if cv[cal.method]["nll"] < cv["none"]["nll"] else "neutral",
               takeaway=f"dev CV NLL none {cv['none']['nll']:.4f} -> {cal.method} {cv[cal.method]['nll']:.4f}; ECE {cv['none']['ece']:.4f} -> {cv[cal.method]['ece']:.4f}")
