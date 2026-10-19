import time, numpy as np, sys
sys.path.insert(0, ".")
from common import *
import mh
p = mh.OnnxMH(sys.argv[1], sys.argv[2], threads=4)
dev = protocol.dev(T)
tt, tr = [], []
for i in range(60):
    s = dev[i].state
    t0 = time.perf_counter(); ids = np.asarray([p.enc.ids(s)], dtype=np.int64); t1 = time.perf_counter()
    p.sess.run(["probs"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}); t2 = time.perf_counter()
    tt.append((t1 - t0) * 1000); tr.append((t2 - t1) * 1000)
print("tok p50/p95", np.percentile(tt, [50, 95]), "run p50/p95", np.percentile(tr, [50, 95]), "len", ids.shape)
ids = np.random.randint(1000, 20000, (1, 128)).astype(np.int64)
r = []
for i in range(30):
    t0 = time.perf_counter(); p.sess.run(["probs"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}); r.append((time.perf_counter() - t0) * 1000)
print("random ids run", np.percentile(r, [50, 95]))
