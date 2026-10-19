"""CPU int8 ONNX latency of an untrained multi-head gte-modernbert with n layers at fixed sequence lengths (batch 1, 4 threads)."""
import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
import mh
OUT = Path(__file__).resolve().parents[2] / "data/v2/helpsteer2/models/probe"
for nl in [int(x) for x in sys.argv[1:]]:
    d = OUT / f"mh_L{nl}"
    if not (d / "model.int8.onnx").exists():
        m, tok = mh.build(nl); mh.save(m, tok, mh.Encoder(tok), d); mh.export(d)
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    s = ort.InferenceSession(str(d / "model.int8.onnx"), o, providers=["CPUExecutionProvider"])
    for L in [128, 192, 256, 320, 384, 512]:
        ids = np.random.randint(1000, 20000, (1, L)).astype(np.int64); am = np.ones_like(ids)
        ts = []
        for i in range(43):
            t0 = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": am}); ts.append((time.perf_counter() - t0) * 1000)
        ts = ts[3:]
        print(f"layers={nl} L={L}: p50={np.percentile(ts,50):.1f} p95={np.percentile(ts,95):.1f}", flush=True)
