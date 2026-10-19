"""int8 dynamic quantization restricted to the weight MatMuls of the lean graph (not attention / RoPE matmuls)."""
import re, sys
from pathlib import Path
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_weights_only(src, dst, exclude_pat=None, per_channel=True):
    m = onnx.load(str(src))
    inits = {i.name for i in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        if not any(x in inits for x in n.input[1:2]):
            continue  # activation x activation (attention)
        if re.search(r"^/MatMul", n.name):
            continue  # RoPE rotation (q @ P)
        if exclude_pat and re.search(exclude_pat, n.name):
            continue
        nodes.append(n.name)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=per_channel, nodes_to_quantize=nodes,
                     extra_options={"MatMulConstBOnly": True})
    return nodes


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else None
    print(len(quantize_weights_only(src, dst, pat)), "nodes quantized")
