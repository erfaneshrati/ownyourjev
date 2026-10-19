"""Predict eval_inputs with the frozen int8 ONNX model (no labels touched)."""
import sys, numpy as np
from pathlib import Path
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor
d = Path(sys.argv[1]); p = OnnxPredictor(d, threads=4)
ev = P.eval_inputs("banking77")
probs = np.concatenate([p.probs([e.state for e in ev[i:i+64]]) for i in range(0, len(ev), 64)])
out = Path("benchmarks/results_v2/banking77/eval_probs.npy"); np.save(out, probs); print(probs.shape, out)
