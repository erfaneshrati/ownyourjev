"""Selective dynamic int8 quantisation: keep activation x activation attention matmuls (and optionally other nodes) in fp32."""
import re
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

def quantize_selective(src, dst, mode="A", per_channel=True):
    m = onnx.load(str(src))
    names = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
    excl = [n for n in names if re.search(r"attn/MatMul(_1)?$", n) or "head" in n]
    if "B" in mode:
        excl += [n for n in names if "mlp/Wo" in n]
    if "C" in mode:
        excl += [n for n in names if re.search(r"layers\.(20|21)/", n)]
    if "D" in mode:
        excl += [n for n in names if re.search(r"layers\.(0|1)/", n)]
    excl = sorted(set(excl))
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=per_channel,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    return dst
