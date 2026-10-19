import sys
from common import *
from ownyourjev.data import read_jsonl
dev = P.dev(T); g = [e.group for e in dev]
L = {e.uid: e for e in read_jsonl(DATA / "dev_labels/g1x5.jsonl")}
Vs = [{e.uid: e for e in read_jsonl(DATA / f)} for f in sys.argv[1:]]
t = {d: np.array([np.dot(L[e.uid].meta["soft"][d], np.arange(5)) for e in dev]) for d in DIMS}
v = np.array([np.mean([V[e.uid].meta["verify"]["v_mean"] for V in Vs if V[e.uid].meta.get("verify")] or [np.nan]) for e in dev]); v[np.isnan(v)] = np.nanmean(v)
print("teacher x5", {d: round(group_spearman([e.labels[d] for e in dev], t[d], g), 3) for d in DIMS})
print("verify", {d: round(group_spearman([e.labels[d] for e in dev], v, g), 3) for d in DIMS})
for w in (0.5, 1, 2, 3, 4):
    print("x5 + w*4*verify, w=", w, {d: round(group_spearman([e.labels[d] for e in dev], t[d] + w * 4 * v, g), 3) for d in DIMS})
print("teacher-vs-verify", round(group_spearman(t["consistency"], v, g), 3))
from scipy.stats import rankdata
def grank(x):
    out = np.zeros(len(x))
    for gg in set(g):
        idx = [i for i in range(len(g)) if g[i] == gg]; out[idx] = rankdata(x[idx]) / len(idx)
    return out
for w in (0.3, 0.5, 1.0):
    print("rank blend", w, {d: round(group_spearman([e.labels[d] for e in dev], grank(t[d]) + w * grank(v), g), 3) for d in DIMS})
import collections
print(collections.Counter(round(x, 2) for x in v).most_common(8))
