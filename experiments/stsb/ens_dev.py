import sys, itertools
from common import *
tags = sys.argv[1:]
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
S = {}
for t in tags:
    m = {e.uid: expected(e) for e in load_jsonl(D / f"labels_dev_{t}.jsonl")}
    S[t] = np.array([m.get(e.uid, np.nan) for e in dev])
h = np.arange(200) % 2 == 0
for r in range(1, 4):
    for c in itertools.combinations(tags, r):
        s = np.nanmean([S[t] for t in c], 0)
        print(f"{'+'.join(c)}: {sp(g, s):.4f} A={sp(g[h], s[h]):.4f} B={sp(g[~h], s[~h]):.4f}")
