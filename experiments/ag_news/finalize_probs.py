"""Compute dev accuracy and eval_inputs probs with the deployed ONNX predictor of a final model dir.
usage: finalize_probs.py TASK MODEL_DIR  -> benchmarks/results_v2/TASK/{eval_probs.npy, dev_probs_deployed.npy, dev_deployed.json}"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
from ownyourjev.evaluate import evaluate
T, md = sys.argv[1], Path(sys.argv[2])
D, R = paths(T)
q = protocol.task_spec(T).parsed["answer"]
pred = OnnxPredictor(md, threads=8)
def run(exs):
    return np.concatenate([pred.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])
dev = protocol.dev(T); golds = [e.labels["answer"] for e in dev]
pd = run(dev)
m = evaluate(q, golds, pd)
r = {"model_dir": str(md), "onnx": str(pred.sess._model_path) if hasattr(pred.sess, "_model_path") else None,
     "dev_accuracy": m["accuracy"], "dev_macro_f1": m.get("macro_f1"),
     "dev_A": evaluate(q, golds[0::2], pd[0::2])["accuracy"], "dev_B": evaluate(q, golds[1::2], pd[1::2])["accuracy"]}
pe = run(protocol.eval_inputs(T))
np.save(R / "eval_probs.npy", pe); np.save(R / "dev_probs_deployed.npy", pd)
r["eval_shape"] = list(pe.shape)
(R / "dev_deployed.json").write_text(json.dumps(r, indent=1))
print("FINAL", json.dumps(r))
