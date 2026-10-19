"""Partial dynamic int8 quantisation (exclude sensitive MatMuls), checked on dev + silver val."""
import os, sys, time, shutil
from pathlib import Path
import numpy as np, onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from ownyourjev.export import OnnxPredictor
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

src = Path("data/v2/clinc150/models/E_lr5")
names = [n.name for n in onnx.load(str(src / "model.onnx")).graph.node if n.op_type in ("MatMul", "Gemm")]
weights = [n for n in names if not n.endswith("attn/MatMul") and not n.endswith("attn/MatMul_1")]
V = {
    "no_mlpWo": [n for n in names if "mlp/Wo" in n or "head" in n],
    "only_Wi": [n for n in names if "mlp/Wi" not in n],
    "only_Wi_Wqkv": [n for n in names if "mlp/Wi" not in n and "Wqkv" not in n],
    "skip_first_last2": [n for n in names if any(f"layers.{i}/" in n for i in (0, 1, 20, 21)) or "head" in n],
}
tr, va = split_pool(load_pool()); dev = P.dev(TASK); ex = dev + va
g = [e.labels["answer"] for e in ex]
pt = np.concatenate([np.load(src / "dev_probs.npy"), np.load(src / "val_probs.npy")])
texts = [e.state for e in dev]
for name in sys.argv[1:] or V:
    d = Path(f"data/v2/clinc150/models/E_lr5_q_{name}"); d.mkdir(exist_ok=True)
    for f in ["backbone", "head.pt", "meta.json"]:
        if not (d / f).exists(): os.symlink((src / f).resolve(), d / f)
    quantize_dynamic(str(src / "model.onnx"), str(d / "model.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=V[name])
    pr = OnnxPredictor(d, threads=4)
    p = np.concatenate([pr.probs([e.state for e in ex[i:i + 64]]) for i in range(0, len(ex), 64)])
    np.save(d / "devval_probs.npy", p)
    ts = []
    for i in range(103):
        t0 = time.perf_counter(); pr.probs([texts[i % len(texts)]]); ts.append((time.perf_counter() - t0) * 1000)
    m = metrics(apply_threshold(p, 0.4), g)
    print(f"{name}: agree={(p.argmax(1) == pt.argmax(1)).mean():.4f} ins={m['ins_acc']:.4f} oos={m['oos_recall']:.3f} p50={np.percentile(ts[3:],50):.1f} p95={np.percentile(ts[3:],95):.1f}", flush=True)
