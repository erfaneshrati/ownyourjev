"""Export a model dir to ONNX + int8, check dev/val agreement after quantisation. Usage: export_check.py <model_dir>"""
import sys, json
from pathlib import Path
import numpy as np
from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

d = Path(sys.argv[1])
if not (d / "model.onnx").exists():
    export_onnx(d)
if not (d / "model.int8.onnx").exists():
    quantize_int8(d / "model.onnx")
dev = P.dev(TASK)
gd = [e.labels["answer"] for e in dev]
pt = np.load(d / "dev_probs.npy")
res = {}
for f in ["model.onnx", "model.int8.onnx"]:
    pr = OnnxPredictor(d, d / f, threads=4)
    p = np.concatenate([pr.probs([e.state for e in dev[i:i + 50]]) for i in range(0, len(dev), 50)])
    np.save(d / f"dev_probs_{f.replace('.', '_')}.npy", p)
    res[f] = {"dev_acc": metrics(p, gd)["acc"], "agree_torch": float((p.argmax(1) == pt.argmax(1)).mean())}
print(json.dumps(res))
