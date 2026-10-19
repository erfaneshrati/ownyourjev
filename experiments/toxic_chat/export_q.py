"""Export model dir to ONNX fp32 + selective int8 (mlp/Wo kept fp32) as model.int8.onnx; compare dev probs vs torch."""
import sys, numpy as np
from common import *
from mh import export_onnx, onnx_probs, texts_of
from quant import quantize_selective
name = sys.argv[1]; md = D / "models" / name
export_onnx(md, int8=False)
quantize_selective(md / "model.onnx", md / "model.int8.onnx", "AB")
dev = protocol.dev(T)
pt = np.load(D / "runs" / name / "dev_probs.npy")
po = onnx_probs(md, texts_of(dev), "model.int8.onnx", threads=4)
np.save(D / "runs" / name / "dev_probs_int8.npy", po)
print("max|diff|", np.abs(po - pt).max(0).round(3), "mean", np.abs(po - pt).mean(0).round(4))
for i, h in enumerate(HEADS):
    y = [e.gold[h] for e in dev]
    print(report(y, po[:, i], f"int8:{h}"), f"cvF1={cv_threshold_f1(y, po[:, i]):.3f}")
