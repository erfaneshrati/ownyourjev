import sys; sys.path.insert(0, ".")
import numpy as np
from common import *
P = np.load(sys.argv[1]); X = P @ np.arange(5)
dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
z = (X - X.mean(0)) / X.std(0)
def r(v, j): return sp(v, G[:, j])
print("raw", [round(r(z[:, j], j), 3) for j in range(5)])
hc = (z[:, 0] + z[:, 1]) / 2
print("h=c=avg", round(r(hc, 0), 3), round(r(hc, 1), 3))
for w in [0.25, 0.5, 1.0]:
    print("coh + w*hc", w, round(r(z[:, 2] + w * hc, 2), 3))
