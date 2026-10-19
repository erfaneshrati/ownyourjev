import sys; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
import numpy as np
from common import *
import mh
d = sys.argv[1]; dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
R = {}
for n in sys.argv[2:]:
    p = mh.OnnxMH(d, n, threads=4); E = np.stack([p.probs(e.state) for e in dev])[:, :5] @ np.arange(5); R[n] = E
    rs = [sp(E[:, j], G[:, j]) for j in range(5)]
    ag = [np.corrcoef(E[:, j], R[sys.argv[2]][:, j])[0, 1] for j in range(5)]
    print(n, "dev", " ".join(f"{r:.3f}" for r in rs), f"mean {np.mean(rs):.3f}", "agree", f"{np.mean(ag):.4f}", flush=True)
