import sys, itertools
from common import *
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
P = {t: np.load(D / "runs" / t / "dev_probs.npy") @ np.arange(6) for t in sys.argv[1:]}
for t in P: print(t, round(sp(g, P[t]), 4))
for r in (2, 3, len(P)):
    for c in itertools.combinations(P, r):
        print("+".join(c), round(sp(g, np.mean([P[t] for t in c], 0)), 4))
