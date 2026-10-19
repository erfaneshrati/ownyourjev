"""Deployed-model predictions (int8 ONNX, batch 1 like serving) for eval_inputs, one array per head."""
import sys, numpy as np
from common import *
from mh import onnx_probs, texts_of
name = sys.argv[1]
ev = protocol.eval_inputs(T)
p = onnx_probs(D / "models" / name, texts_of(ev), "model.int8.onnx", threads=8)
out = D / "runs" / name
for i, h in enumerate(HEADS):
    np.save(out / f"eval_probs_{h}.npy", p[:, i].astype(np.float64))
print(p.shape, (p >= 0.5).mean(0))
