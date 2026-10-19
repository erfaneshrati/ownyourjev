"""Selective dynamic int8 quantisation variants (library quantize_int8 also quantises the activation x activation
attention MatMuls); check dev Spearman of each with onnxruntime."""
import sys, onnx
from common import *
from onnxruntime.quantization import QuantType, quantize_dynamic
from ownyourjev.export import OnnxPredictor
md = Path(sys.argv[1])
names = [n.name for n in onnx.load(str(md / "model.onnx")).graph.node if n.op_type in ("MatMul", "Gemm")]
act = [n for n in names if n.endswith("attn/MatMul") or n.endswith("attn/MatMul_1")]
V = {
  "q_noact": act + ["/m/head/Gemm"],
  "q_noact_nomlpwo": act + ["/m/head/Gemm"] + [n for n in names if "mlp/Wo" in n],
}
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
for v, excl in V.items():
    out = md / f"model.{v}.onnx"
    if not out.exists():
        quantize_dynamic(str(md / "model.onnx"), str(out), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    p = OnnxPredictor(md, out, threads=4)
    a = np.concatenate([p.probs([e.state]) for e in dev])
    np.save(md / f"dev_probs_{v}.npy", a)
    print(v, "dev spearman", round(sp(g, a @ np.arange(6)), 4), "maxdiff vs fp32", float(np.abs(a - np.load(md / "dev_probs_model.onnx.npy")).max()))
