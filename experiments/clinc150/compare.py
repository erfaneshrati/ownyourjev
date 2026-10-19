"""Pooled (dev + silver val) comparison: in-scope acc, oos recall, weighted by the 18% test oos share."""
import sys
import numpy as np
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *
pool = load_pool(); tr, va = split_pool(pool)
dev = P.dev(TASK)
g = [e.labels["answer"] for e in dev] + [e.labels["answer"] for e in va]
for n in sys.argv[1:]:
    d = f"data/v2/clinc150/models/{n}"
    p = np.concatenate([np.load(f"{d}/dev_probs.npy"), np.load(f"{d}/val_probs.npy")])
    row = []
    for tau in [0.0, 0.2, 0.3, 0.4, 0.5]:
        m = metrics(apply_threshold(p, tau), g)
        row.append(f"t{tau}: ins={m['ins_acc']:.4f} oos={m['oos_recall']:.3f} w={m['weighted']:.4f}")
    print(f"{n:12s} " + " | ".join(row))
