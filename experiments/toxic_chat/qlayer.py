"""Per-layer sensitivity: quantise everything (except attn act matmuls/head) but keep mlp/Wo of ONE layer in fp32... reversed:
quantise mlp/Wo of only one layer k (others fp32) and measure dev drift, to find the outlier layers."""
import re, sys, numpy as np, onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from common import *
from mh import onnx_probs, texts_of
name = sys.argv[1]
md = D / "models" / name
dev = protocol.dev(T); tx = texts_of(dev)
pt = np.load(D / "runs" / name / "dev_probs.npy")
m = onnx.load(str(md / "model.onnx"))
names = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
for k in range(22):
    excl = [n for n in names if ("mlp/Wo" in n and f"layers.{k}/" not in n) or "head" in n]
    quantize_dynamic(str(md / "model.onnx"), str(md / "model.qtmp.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    po = onnx_probs(md, tx, "model.qtmp.onnx", threads=4)
    print(k, "max", np.abs(po - pt).max().round(3), "mean", np.abs(po - pt).mean().round(4), flush=True)
