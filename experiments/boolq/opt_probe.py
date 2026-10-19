import sys
from pathlib import Path
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import QuantType, quantize_dynamic
d = Path(sys.argv[1])
m = optimizer.optimize_model(str(d / "model.onnx"), model_type="bert", num_heads=12, hidden_size=768, opt_level=1)
print(m.get_fused_operator_statistics())
m.save_model_to_file(str(d / "model.opt.onnx"))
quantize_dynamic(str(d / "model.opt.onnx"), str(d / "model.opt.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
quantize_dynamic(str(d / "model.onnx"), str(d / "model.int8nopc.onnx"), weight_type=QuantType.QInt8, per_channel=False, op_types_to_quantize=["MatMul", "Gemm"])
quantize_dynamic(str(d / "model.onnx"), str(d / "model.int8cb.onnx"), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], extra_options={"MatMulConstBOnly": True})
