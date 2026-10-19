"""Quantisation variants for a trained model dir: dev accuracy + quick CPU latency (4 threads) for each ONNX file."""
import json, time, sys
from pathlib import Path
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor
from ownyourjev.evaluate import evaluate

def run(d, files):
    d = Path(d); dev = P.dev("banking77"); q = P.task_spec("banking77").parsed["answer"]
    gold = [e.labels["answer"] for e in dev]
    for f in files:
        p = OnnxPredictor(d, onnx_path=d / f, threads=4)
        pr = np.concatenate([p.probs([e.state]) for e in dev])
        t = []
        for e in dev[:100]:
            t0 = time.perf_counter(); p.probs([e.state]); t.append((time.perf_counter() - t0) * 1000)
        print(json.dumps({"file": f, "dev_acc": evaluate(q, gold, pr)["accuracy"], "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95))}))

if __name__ == "__main__":
    d = sys.argv[1]
    from onnxruntime.quantization import QuantType, quantize_dynamic
    if "pt" in sys.argv[2:]:
        quantize_dynamic(str(Path(d) / "model.onnx"), str(Path(d) / "model.int8pt.onnx"), weight_type=QuantType.QInt8, per_channel=False, op_types_to_quantize=["MatMul", "Gemm"])
    run(d, [f for f in sys.argv[2:] if f.endswith(".onnx")] + (["model.int8pt.onnx"] if "pt" in sys.argv[2:] else []))
