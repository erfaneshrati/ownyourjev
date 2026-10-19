import sys, itertools; sys.path.insert(0, "experiments/summeval")
from common import *
from ownyourjev.data import read_jsonl
d = P.dev(T); uid = [e.uid for e in d]
files = sorted((DATA / "dev_labels").glob("*.jsonl"))
S = {}
for f in files:
    ex = {e.uid: e for e in read_jsonl(f)}
    S[f.stem] = {k: np.array([np.dot(ex[u].meta["soft"][k], np.arange(5)) if ex[u].labels else np.nan for u in uid]) for k in DIMS}
def z(v):  # within-doc ranks not needed; nan->mean
    v = v.copy(); v[np.isnan(v)] = np.nanmean(v); return v
names = sys.argv[1:] or list(S)
for r in (1, 2, 3):
    for combo in itertools.combinations(names, r):
        pred = {k: np.mean([z(S[c][k]) for c in combo], 0) for k in DIMS}
        res = score_dev(pred, d)
        print(f"{res['mean']:.3f}", {k: round(res[k], 3) for k in DIMS}, " + ".join(c.split("__")[0] + ":" + c.split("_")[-1] for c in combo))
