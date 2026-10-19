import sys; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
import numpy as np
from common import *
from postproc import blend_hc
dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
for f in sys.argv[1:]:
    P = np.load(f)[:, :5]
    for nm, Q in [("raw", P), ("blend", blend_hc(P))]:
        E = Q @ np.arange(5); rs = [sp(E[:, j], G[:, j]) for j in range(5)]
        print(f.split("/")[-2], nm, " ".join(f"{r:.3f}" for r in rs), f"mean {np.mean(rs):.3f}", " A/B halves:", f"{np.mean([sp(E[:100, j], G[:100, j]) for j in range(5)]):.3f}", f"{np.mean([sp(E[100:, j], G[100:, j]) for j in range(5)]):.3f}")
