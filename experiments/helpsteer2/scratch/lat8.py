import sys, time; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
import numpy as np, onnxruntime as ort
d = sys.argv[1]
o = ort.SessionOptions(); o.intra_op_num_threads = 8
s = ort.InferenceSession(f"{d}/model.int8.onnx", o, providers=["CPUExecutionProvider"])
R = {}
for rep in range(40):
    for L in [128, 160, 192, 224, 256]:
        ids = np.random.randint(1000, 20000, (1, L)).astype(np.int64)
        t0 = time.perf_counter(); s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}); R.setdefault(L, []).append((time.perf_counter() - t0) * 1000)
for L, t in R.items(): print(L, "p50 %.1f p95 %.1f" % (np.percentile(t[2:], 50), np.percentile(t[2:], 95)))
