"""Export a MultiHead model to ONNX (fp32; int8 with --int8), report dev metrics per runtime, optionally write eval probs.
Usage: export_eval.py NAME [--int8] [--eval]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, TASK, dev_report
from mt_model import export_onnx, onnx_probs
from benchmarks.v2 import protocol as P
from ownyourjev.export import quantize_int8
d = DATA / "models" / sys.argv[1]
if not (d / "model.onnx").exists():
    export_onnx(d)
dev = P.dev(TASK); texts = [e.state["text"] for e in dev]
runs = {"fp32onnx": "model.onnx"}
if "--int8" in sys.argv:
    if not (d / "model.int8.onnx").exists():
        quantize_int8(d / "model.onnx")
    runs["int8"] = "model.int8.onnx"
for k, f in runs.items():
    p = onnx_probs(d, texts, f); np.save(d / f"dev_probs_{k}.npy", p)
    r = dev_report(p[:, 0], p[:, 1:], dev)
    print(k, {kk: round(v, 4) for kk, v in r.items()})
if "--eval" in sys.argv:
    ev = onnx_probs(d, [e.state["text"] for e in P.eval_inputs(TASK)], "model.onnx")
    np.save(d / "eval_probs_fp32onnx.npy", ev); print("eval", ev.shape)
