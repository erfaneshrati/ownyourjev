"""Partial dynamic int8 for UltraBERT: exclude activation x activation MatMuls (attention scores / values) and optionally others."""
import sys, onnx
from common import *
from onnxruntime.quantization import QuantType, quantize_dynamic
T = "paws"
d = ROOT / "data/v2/paws/models" / sys.argv[1]
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
m = onnx.load(str(d / "model.onnx"), load_external_data=False)
mm = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
VARS = {
    "noattn": lambda n: n.endswith("self_attn/MatMul") or n.endswith("self_attn/MatMul_1"),
    "noattn_nodown": lambda n: n.endswith("self_attn/MatMul") or n.endswith("self_attn/MatMul_1") or "down_proj" in n,
}
for v in sys.argv[2].split(","):
    excl = [n for n in mm if VARS[v](n)] + ["/m/head/Gemm"]
    out = d / f"model.int8_{v}.onnx"
    quantize_dynamic(str(d / "model.onnx"), str(out), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    p = onnx_probs(d, dev, onnx_path=out)
    np.save(d / f"dev_probs_int8_{v}.npy", p)
    print(v, "excluded", len(excl), "dev acc", acc_f1(y, p)[0], flush=True)
