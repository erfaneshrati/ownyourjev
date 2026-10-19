import sys, itertools
from common import *
dev = protocol.dev(T)
names = sys.argv[1:]
L = {n: {e.uid: e for e in load_jsonl(LD / f"dev_{n}.jsonl")} for n in names}
def vec(n, a): return np.array([expected(L[n][x.uid], a) for x in dev])
G = {a: np.array([x.gold[a] for x in dev]) for a in A}
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        rs = [sp(np.mean([vec(n, a) for n in combo], 0), G[a]) for a in A]
        print(f"{'+'.join(combo):60s} " + " ".join(f"{v:.3f}" for v in rs) + f"  mean {np.mean(rs):.3f}")
