import sys, itertools, json
import numpy as np
sys.path.insert(0, "experiments/toxigen")
from common import LAB, TASK, dev_report, load, soft_of
from benchmarks.v2 import protocol as P
dev = P.dev(TASK)
names = sys.argv[1].split(",")
R = {n: load(LAB / f"dev_{n}.jsonl") for n in names}
def arr(n):
    s = [soft_of(R[n][e.uid]) for e in dev]
    return np.array([x[0] for x in s]), np.stack([x[1] for x in s])
A = {n: arr(n) for n in names}
for k in range(1, len(names) + 1):
    for c in itertools.combinations(names, k):
        p = np.mean([A[n][0] for n in c], 0); d = np.mean([A[n][1] for n in c], 0)
        r = dev_report(p, d, dev); ra = dev_report(p, d, dev, np.arange(200) % 2 == 0); rb = dev_report(p, d, dev, np.arange(200) % 2 == 1)
        print("+".join(c), {kk: round(r[kk], 3) for kk in ["acc_es@2", "acc_es_best", "thr_es", "acc_comb_best", "spearman_es", "auroc_es"]}, "A", ra["acc_es_best"], "B", rb["acc_es_best"])
