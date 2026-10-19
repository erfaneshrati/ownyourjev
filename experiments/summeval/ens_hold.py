import sys, numpy as np
from common import *
dev = P.dev(T)
ps = [np.load(DATA / "dev_hold" / f"{n}.npy") for n in sys.argv[1:]]
for p, n in zip(ps, sys.argv[1:]):
    print(n, round(score_dev({d: p[:, j] @ np.arange(5) for j, d in enumerate(DIMS)}, dev)["mean"], 4))
p = np.mean(ps, 0)
r = score_dev({d: p[:, j] @ np.arange(5) for j, d in enumerate(DIMS)}, dev)
print("ENS", {k: round(v, 4) for k, v in r.items()})
