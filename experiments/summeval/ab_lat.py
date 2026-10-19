import sys, time, numpy as np, onnxruntime as ort
from pathlib import Path
from common import *
import export_mh as X
d = Path(sys.argv[1]); names = sys.argv[2:]
pr = X.Predictor(d, threads=4, int8=True)
sess = {}
for n in names:
    o = ort.SessionOptions(); o.intra_op_num_threads = 4
    sess[n] = ort.InferenceSession(str(d / f"{n}.onnx"), o, providers=["CPUExecutionProvider"])
dev = P.dev(T); t = {n: [] for n in names}
for i in range(120):
    for n in names:
        pr.sess = sess[n]; t0 = time.perf_counter(); pr.probs([dev[i % len(dev)].state]); t[n].append((time.perf_counter() - t0) * 1000)
for n in names:
    c = t[n][5:]; print(n, "p50", round(np.percentile(c, 50), 1), "p95", round(np.percentile(c, 95), 1))
