"""Export a saved model to ONNX + int8, check dev accuracy after quantisation, write eval probs (int8)."""
import sys, json; sys.path.insert(0,"experiments/sst5")
from pathlib import Path
import numpy as np
from lab import DEV, DEVY, P, T, report
from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor
d = Path(sys.argv[1])
if not (d / "model.onnx").exists(): export_onnx(d)
if not (d / "model.int8.onnx").exists(): quantize_int8(d / "model.onnx")
out = {}
for kind, f in [("fp32", "model.onnx"), ("int8", "model.int8.onnx")]:
    pr = OnnxPredictor(d, onnx_path=d / f, threads=8)
    pd = np.concatenate([pr.probs([e.state for e in DEV[i:i+32]]) for i in range(0, len(DEV), 32)])
    out[kind] = report(f"{d.name}-{kind}", pd)
    np.save(d / f"dev_probs_{kind}.npy", pd)
    if kind == "int8" and "--eval" in sys.argv:
        ev = P.eval_inputs(T)
        pe = np.concatenate([pr.probs([e.state for e in ev[i:i+32]]) for i in range(0, len(ev), 32)])
        np.save(d / "eval_probs_int8.npy", pe); print("eval probs", pe.shape)
json.dump(out, open(d / "onnx_dev.json", "w"))
