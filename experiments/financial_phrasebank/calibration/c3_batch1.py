"""Batch-size-1 predictions of the deployed int8 model (same artifact/preprocessing/threads as finalize_probs.py, one text
per call), dev + eval_inputs (no labels). Refit calibrator (none/temperature) on batch-1 dev; write BATCH1.json."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ag_news"))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator, cv_scores
T = "financial_phrasebank"
md = ROOT / "data/v2/financial_phrasebank/models/v1seeded_s1"
R = ROOT / "benchmarks/results_v2" / T
pred = OnnxPredictor(md, threads=8)
run1 = lambda exs: np.concatenate([pred.probs([e.state]) for e in exs])
q = protocol.task_spec(T).parsed["answer"]
dev = protocol.dev(T); y = np.array([q.options.index(e.labels["answer"]) for e in dev])
pd = run1(dev); np.save(R / "dev_probs_b1.npy", pd)
pdb = np.load(R / "dev_probs_deployed.npy")
print("dev b1 acc", float((pd.argmax(1) == y).mean()), "batched", float((pdb.argmax(1) == y).mean()),
      "dev flips", int((pd.argmax(1) != pdb.argmax(1)).sum()), "max|dp|", float(np.abs(pd - pdb).max()))
pe = run1(protocol.eval_inputs(T)); np.save(R / "eval_probs_b1.npy", pe)
pf = np.load(R / "eval_probs.npy")
print("test flips (no labels)", int((pe.argmax(1) != pf.argmax(1)).sum()), "/", len(pe), "max|dp|", float(np.abs(pe - pf).max()))
# determinism: re-run a few at batch 1
print("b1 repeat max|diff|", float(np.abs(run1(dev[:20]) - pd[:20]).max()))
cv_all = cv_scores(pd, y)
for m, s in cv_all.items():
    print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} conf {s['mean_conf']:.3f}")
cal, cv = select_calibrator(pd, y, methods=("none", "temperature"))
print("chosen", cal.method, cal.params)
c = cal.to_dict(); c["dev_cv"] = cv_all; c["n_fit"] = len(y)
out = {"description": "Deployed model.int8.onnx via OnnxPredictor(threads=8), max_len 96, ONE text per call (batch size 1), "
       "dev and eval_inputs; otherwise identical to finalize_probs.py. Calibrator refit on batch-1 dev (out-of-sample: dev uids "
       "never trained on) by 5-fold CV NLL over argmax-preserving none/temperature; dev_cv also lists vector/isotonic for reference.",
       "heads": {"answer": {"probs": "benchmarks/results_v2/financial_phrasebank/eval_probs_b1.npy",
                            "dev_probs": "benchmarks/results_v2/financial_phrasebank/dev_probs_b1.npy"}},
       "calibration": {"answer": c}}
(R / "BATCH1.json").write_text(json.dumps(out, indent=1, default=float))
