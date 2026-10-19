import sys
import numpy as np
from common import *
import mh, stacking
d = sys.argv[1]
dev = protocol.dev(T)
f = Path(d) / "dev_probs.npy"
if f.exists():
    P = np.load(f)
else:
    p = mh.OnnxMH(d, sys.argv[2] if len(sys.argv) > 2 else "model.onnx"); P = np.stack([p.probs(e.state) for e in dev]); np.save(f, P)
G = np.array([[e.gold[a] for a in A] for e in dev], float)
for alpha in [1, 10, 100, 1000]:
    b, c = stacking.cv(P, G, alpha)
    print(f"alpha {alpha:5}: raw " + " ".join(f"{x:.3f}" for x in b) + f" ({b.mean():.3f})  stacked-CV " + " ".join(f"{x:.3f}" for x in c) + f" ({c.mean():.3f})")
