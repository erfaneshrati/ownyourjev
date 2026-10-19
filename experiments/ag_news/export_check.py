"""Export a trained model to ONNX + int8, check dev accuracy of the int8 model, optionally write eval probs.
usage: export_check.py TASK MODEL_NAME [--eval]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor
from ownyourjev.evaluate import evaluate

T, name = sys.argv[1], sys.argv[2]
D, R = paths(T)
md = D / "models" / name
if not (md / "model.int8.onnx").exists():
    export_onnx(md)
    quantize_int8(md / "model.onnx")
q = protocol.task_spec(T).parsed["answer"]
pred = OnnxPredictor(md, threads=8)
def run(exs, bs=64):
    out = []
    for i in range(0, len(exs), bs):
        out.append(pred.probs([e.state for e in exs[i:i + bs]]))
    return np.concatenate(out)
dev = protocol.dev(T)
p8 = run(dev)
np.save(md / "dev_probs_int8.npy", p8)
golds = [e.labels["answer"] for e in dev]
pf = np.load(md / "dev_probs.npy")
r = {"name": name, "dev_acc_torch": evaluate(q, golds, pf)["accuracy"], "dev_acc_int8": evaluate(q, golds, p8)["accuracy"],
     "argmax_agree": float((pf.argmax(1) == p8.argmax(1)).mean())}
if "--eval" in sys.argv:
    pe = run(protocol.eval_inputs(T))
    np.save(md / "eval_probs_int8.npy", pe)
    r["eval_saved"] = pe.shape
print("EXPORT", json.dumps(r, default=str))
