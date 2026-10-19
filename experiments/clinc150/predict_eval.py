"""Export the frozen model to ONNX fp32, check vs torch on dev, predict eval_inputs, apply the oos threshold."""
import sys, json
from pathlib import Path
import numpy as np
from ownyourjev.export import export_onnx, OnnxPredictor
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

d = Path(sys.argv[1]); tau = float(sys.argv[2])
if not (d / "model.onnx").exists():
    export_onnx(d)
onnx_file = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"
pr = OnnxPredictor(d, d / onnx_file, threads=8)
def run(exs): return np.concatenate([pr.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])
dev = P.dev(TASK); gd = [e.labels["answer"] for e in dev]
pd = run(dev); pt = np.load(d / "dev_probs.npy")
print("onnx vs torch dev argmax agree", float((pd.argmax(1) == pt.argmax(1)).mean()), "max abs diff", float(np.abs(pd - pt).max()))
print("dev raw", metrics(pd, gd)); print("dev thr", metrics(apply_threshold(pd, tau), gd))
ev = P.eval_inputs(TASK)
pe = apply_threshold(run(ev), tau)
out = Path("data/v2/clinc150"); np.save(out / "eval_probs.npy", pe.astype(np.float32))
print("eval shape", pe.shape, "pred oos share", float((pe.argmax(1) == OOS).mean()))
