"""Partial dynamic int8 quantisation (exclude outlier-prone MatMuls) for UltraBERT; dev check per variant."""
import sys, re, json; sys.path.insert(0,"experiments/sst5")
from pathlib import Path
import numpy as np, onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from lab import DEV, DEVY, report
from ownyourjev.export import OnnxPredictor
d = Path(sys.argv[1]); variants = sys.argv[2].split(",")
m = onnx.load(d / "model.onnx", load_external_data=False)
mm = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
rules = {
 "nodown": lambda n: "down_proj" in n or re.search(r"self_attn/MatMul", n) or "head" in n,
 "projonly_attn": lambda n: not re.search(r"(q|k|v|o)_proj", n),
 "nodown_last4": lambda n: "down_proj" in n or re.search(r"self_attn/MatMul", n) or "head" in n or re.search(r"layers\.(2[0-3])/", n),
 "nodown_first2": lambda n: "down_proj" in n or re.search(r"self_attn/MatMul", n) or "head" in n or re.search(r"layers\.(0|1)/", n),
}
for v in variants:
    excl = [n for n in mm if rules[v](n)]
    out = d / f"model.{v}.onnx"
    quantize_dynamic(str(d / "model.onnx"), str(out), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    pr = OnnxPredictor(d, onnx_path=out, threads=8)
    pd = np.concatenate([pr.probs([e.state for e in DEV[i:i+32]]) for i in range(0, len(DEV), 32)])
    print(v, "excluded", len(excl), "of", len(mm)); report(f"{d.name}-{v}", pd); np.save(d / f"dev_probs_{v}.npy", pd)
