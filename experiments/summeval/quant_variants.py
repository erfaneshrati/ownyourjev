"""int8 variants for a fold model: exclude rotary R matmuls and optionally other sensitive matmuls; report dev + latency."""
import sys, json, time
from pathlib import Path
import numpy as np, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from common import *
import export_mh as X

def nodes(model_path):
    m = onnx.load(str(model_path)); init = {i.name: i for i in m.graph.initializer}
    out = {"rot": [], "attn_out": [], "mlp_out": [], "mlp_in": [], "qkv": [], "other": []}
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm"): continue
        b = n.input[1]
        if b not in init:
            continue
        shp = tuple(init[b].dims)
        if shp == (64, 64): out["rot"].append(n.name)
        elif shp == (768, 2304): out["qkv"].append(n.name)
        elif shp == (768, 768): out["attn_out"].append(n.name)
        elif shp == (1152, 768): out["mlp_out"].append(n.name)
        elif shp == (768, 2304) or shp == (768, 2 * 1152): out["mlp_in"].append(n.name)
        else: out["other"].append(f"{n.name}:{shp}")
    return out

def run(d, name, exclude):
    d = Path(d); q = d / f"model.{name}.onnx"
    quantize_dynamic(str(d / "model.onnx"), str(q), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=exclude)
    pr = X.Predictor(d, threads=4, int8=False); 
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    pr.sess = ort.InferenceSession(str(q), o, providers=["CPUExecutionProvider"])
    dev = P.dev(T)
    t = []; ps = []
    for e in dev:
        t0 = time.perf_counter(); ps.append(pr.probs([e.state])); t.append((time.perf_counter() - t0) * 1000)
    p = np.concatenate(ps)
    r = score_dev({dd: p[:, j] @ np.arange(5) for j, dd in enumerate(DIMS)}, dev)
    print(name, {k: round(v, 4) for k, v in r.items()}, "p50", round(np.percentile(t, 50), 1), "p95", round(np.percentile(t, 95), 1), flush=True)

if __name__ == "__main__":
    d = Path(sys.argv[1]); N = nodes(d / "model.onnx")
    print({k: len(v) for k, v in N.items()}, N["other"])
    run(d, "q_norot", N["rot"])
    run(d, "q_norot_noout", N["rot"] + N["attn_out"] + N["mlp_out"])
    run(d, "q_all", [])
