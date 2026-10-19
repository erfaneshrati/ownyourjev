"""int8 dynamic quantisation variants of a multi-head model; dev Spearman + agreement with fp32 + latency at 128 tokens."""
import sys, time, json
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from common import *
import mh

def names(path):
    m = onnx.load(str(path), load_external_data=False)
    return [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]

def variant_excludes(nm, v, nl):
    act = [n for n in nm if n.endswith("attn/MatMul") or n.endswith("attn/MatMul_1")]
    ex = list(act)
    if "head" in v: ex += [n for n in nm if "heads" in n]
    if "mlpwo" in v: ex += [n for n in nm if "mlp/Wo" in n]
    if "edge" in v: ex += [n for n in nm if any(f"layers.{i}/" in n for i in (0, 1, nl - 2, nl - 1))]
    if "all" in v: ex = []
    return ex

def quantize(d, v):
    d = Path(d); nl = json.loads((d / "meta.json").read_text())["n_layers"]
    nm = names(d / "model.onnx"); out = d / f"model.int8.{v}.onnx"
    quantize_dynamic(str(d / "model.onnx"), str(out), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"],
                     nodes_to_exclude=variant_excludes(nm, v, nl), extra_options={"MatMulConstBOnly": True} if "constb" in v else {})
    return out.name

def evaluate_onnx(d, name, dev, ref=None):
    p = mh.OnnxMH(d, name, threads=4)
    P = np.stack([p.probs(e.state) for e in dev]); es = P @ np.arange(5)
    rs = [sp(es[:, j], [e.gold[x] for e in dev]) for j, x in enumerate(A)]
    ids = np.random.randint(1000, 20000, (1, p.enc.budget)).astype(np.int64); ts = []
    for i in range(33):
        t0 = time.perf_counter(); p.sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}); ts.append((time.perf_counter() - t0) * 1000)
    agree = [float(np.corrcoef(es[:, j], ref[:, j])[0, 1]) for j in range(5)] if ref is not None else None
    return es, rs, float(np.percentile(ts[3:], 50)), agree

if __name__ == "__main__":
    d = sys.argv[1]; dev = protocol.dev(T)
    ref, rs, t, _ = evaluate_onnx(d, "model.onnx", dev)
    print(f"fp32        dev {np.mean(rs):.3f} " + " ".join(f"{x:.3f}" for x in rs) + f" p50 {t:.1f}", flush=True)
    for v in sys.argv[2:]:
        n = quantize(d, v)
        es, rs, t, ag = evaluate_onnx(d, n, dev, ref)
        print(f"{v:12s} dev {np.mean(rs):.3f} " + " ".join(f"{x:.3f}" for x in rs) + f" p50 {t:.1f} agree " + " ".join(f"{x:.3f}" for x in ag), flush=True)
