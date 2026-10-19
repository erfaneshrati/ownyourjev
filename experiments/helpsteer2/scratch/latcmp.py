import sys, time; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
import numpy as np
from common import *
import mh
d = sys.argv[1]; names = sys.argv[2:]
ps = {n: mh.OnnxMH(d, n, threads=4) for n in names}
st = [e.state for e in protocol.dev(T)]
T_ = {n: [] for n in names}
for i in range(203):
    for n, p in ps.items():
        t0 = time.perf_counter(); p.probs(st[i % 200]); T_[n].append((time.perf_counter() - t0) * 1000)
for n, t in T_.items():
    print(n, "p50 %.1f p95 %.1f" % (np.percentile(t[3:], 50), np.percentile(t[3:], 95)))
