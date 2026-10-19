"""Latency (int8 with mlp.Wo fp32, as deployed) of untrained multi-head models by depth and length; interleaved to share machine noise."""
import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
import mh
OUT = Path(__file__).resolve().parents[2] / "data/v2/helpsteer2/models/probe"
sess = {}
for nl in [int(x) for x in sys.argv[1:]]:
    d = OUT / f"mhq_L{nl}"
    if not (d / "model.int8.onnx").exists():
        m, tok = mh.build(nl); mh.save(m, tok, mh.Encoder(tok), d); mh.export(d)
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    sess[nl] = ort.InferenceSession(str(d / "model.int8.onnx"), o, providers=["CPUExecutionProvider"])
res = {}
for rep in range(30):
    for nl, s in sess.items():
        for L in [96, 112, 128, 144, 160, 192]:
            ids = np.random.randint(1000, 20000, (1, L)).astype(np.int64)
            t0 = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
            res.setdefault((nl, L), []).append((time.perf_counter() - t0) * 1000)
for (nl, L), ts in res.items():
    print(f"layers={nl} L={L}: p50={np.percentile(ts[2:],50):.1f} p95={np.percentile(ts[2:],95):.1f}")
