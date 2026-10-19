import sys, numpy as np
from common import *
from mh import onnx_probs, texts_of
from quant import quantize_selective
name = sys.argv[1]; modes = sys.argv[2].split(",")
md = D / "models" / name
dev = protocol.dev(T)
pt = np.load(D / "runs" / name / "dev_probs.npy")
for mode in modes:
    f = f"model.q{mode}.onnx"
    quantize_selective(md / "model.onnx", md / f, mode)
    po = onnx_probs(md, texts_of(dev), f)
    print(mode, "max|diff|", np.abs(po - pt).max(0).round(3), "mean|diff|", np.abs(po - pt).mean(0).round(4))
    for i, h in enumerate(HEADS):
        print("  ", report([e.gold[h] for e in dev], po[:, i], h))
