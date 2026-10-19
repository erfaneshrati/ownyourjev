"""Export a model dir to ONNX (+int8) and compare int8 dev probs to torch dev probs."""
import sys, numpy as np
from common import *
from mh import export_onnx, onnx_probs, texts_of
name = sys.argv[1]
md = D / "models" / name
export_onnx(md)
dev = protocol.dev(T)
pt = np.load(D / "runs" / name / "dev_probs.npy")
for f in ["model.onnx", "model.int8.onnx"]:
    po = onnx_probs(md, texts_of(dev), f)
    np.save(D / "runs" / name / f"dev_probs_{f.replace('.', '_')}.npy", po)
    print(f, "max|diff|", np.abs(po - pt).max(0))
    for i, h in enumerate(HEADS):
        y = [e.gold[h] for e in dev]
        print("  ", report(y, po[:, i], h))
