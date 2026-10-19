import sys, time
import numpy as np, onnxruntime as ort
d = sys.argv[1]; S = {}
for th in [4, 8]:
    o = ort.SessionOptions(); o.intra_op_num_threads = th
    S[th] = ort.InferenceSession(f"{d}/model.int8.onnx", o, providers=["CPUExecutionProvider"])
R = {}
for rep in range(50):
    for L in [128, 192]:
        ids = np.random.randint(1000, 20000, (1, L)).astype(np.int64)
        for th, s in S.items():
            t0 = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}); R.setdefault((th, L), []).append((time.perf_counter() - t0) * 1000)
for k, t in R.items(): print(k, "p50 %.1f p95 %.1f" % (np.percentile(t[2:], 50), np.percentile(t[2:], 95)))
