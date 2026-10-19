import sys, numpy as np
from common import *
dev = P.dev(T); dg = sorted({e.group for e in dev})
for n in sys.argv[1:]:
    p = np.load(DATA / "dev_hold" / f"{n}.npy"); out = []
    for h in (0, 1):
        half = set(dg[h::2]); idx = [i for i, e in enumerate(dev) if e.group not in half]
        r = score_dev({d: p[idx, j] @ np.arange(5) for j, d in enumerate(DIMS)}, [dev[i] for i in idx]); out.append(round(r["mean"], 4))
    print(n, "eval-on-not-half0:", out[0], "eval-on-not-half1:", out[1])
