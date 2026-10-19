import sys, time, numpy as np, onnxruntime as ort
from pathlib import Path
d = Path(sys.argv[1]); name = sys.argv[2]
o = ort.SessionOptions(); o.intra_op_num_threads = 4
s = ort.InferenceSession(str(d / name), o, providers=["CPUExecutionProvider"])
for L in [64, 128, 192, 256, 320, 384]:
    ids = np.random.randint(1000, 20000, size=(1, L)).astype(np.int64); m = np.ones_like(ids)
    for _ in range(3): s.run(None, {"input_ids": ids, "attention_mask": m})
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": m}); ts.append((time.perf_counter() - t0) * 1000)
    print(f"{name} L={L}: p50={np.percentile(ts,50):.1f} p95={np.percentile(ts,95):.1f}", flush=True)
