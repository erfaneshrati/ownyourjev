import itertools, sys, numpy as np
from common import *
names = sys.argv[1:]
L = {n: read_jsonl(DATA/"dev_labels"/f"{n}.jsonl") for n in names}
g = np.array([e.gold[QID] for e in L[names[0]]])
print("gold yes rate", g.mean())
P = {n: np.array([e.labels[QID] for e in L[n]]) for n in names}
for n in names: print(f"{n:30s} acc={np.mean(P[n]==g):.3f}")
for a,b in itertools.combinations(names,2):
    m = P[a]==P[b]
    print(f"{a:28s} {b:28s} agree={m.mean():.3f} acc_when_agree={np.mean(g[m]==P[a][m]):.3f}")
for trio in itertools.combinations(names,3):
    maj = (sum(P[n].astype(int) for n in trio) >= 2)
    print(trio, "majority acc", round(np.mean(maj==g),3))
