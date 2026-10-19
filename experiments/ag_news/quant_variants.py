"""Compare int8 quantisation variants of a model's ONNX export on dev (accuracy and argmax agreement with fp32)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic
from ownyourjev.export import OnnxPredictor
from ownyourjev.evaluate import evaluate
T, name = sys.argv[1], sys.argv[2]
D, _ = paths(T); md = D / "models" / name
q = protocol.task_spec(T).parsed["answer"]
dev = protocol.dev(T); golds = [e.labels["answer"] for e in dev]
def run(path):
    p = OnnxPredictor(md, onnx_path=path, threads=8)
    return np.concatenate([p.probs([e.state for e in dev[i:i + 64]]) for i in range(0, len(dev), 64)])
pf = run(md / "model.onnx")
print("fp32 onnx", evaluate(q, golds, pf)["accuracy"])
for tag, kw in {"pc_u8": dict(per_channel=True, weight_type=QuantType.QUInt8), "nopc_s8": dict(per_channel=False, weight_type=QuantType.QInt8),
                "pc_s8_rr": dict(per_channel=True, weight_type=QuantType.QInt8, reduce_range=True)}.items():
    out = md / f"model.{tag}.onnx"
    if not out.exists():
        quantize_dynamic(str(md / "model.onnx"), str(out), op_types_to_quantize=["MatMul", "Gemm"], **kw)
    p8 = run(out)
    print(tag, evaluate(q, golds, p8)["accuracy"], "agree", float((pf.argmax(1) == p8.argmax(1)).mean()))
