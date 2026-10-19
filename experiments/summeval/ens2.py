import sys
from common import *
from ownyourjev.data import read_jsonl
d = P.dev(T); uid = [e.uid for e in d]
def load(n):
    ex = {e.uid: e for e in read_jsonl(DATA / "dev_labels" / f"{n}.jsonl")}
    return {k: np.array([np.dot(ex[u].meta["soft"][k], np.arange(5)) if ex[u].labels else np.nan for u in uid]) for k in DIMS}
for combo in sys.argv[1:]:
    parts = combo.split("+"); S = [load(p) for p in parts]
    pred = {k: np.nanmean([s[k] for s in S], 0) for k in DIMS}
    r = score_dev(pred, d); print(f"{r['mean']:.3f}", {k: round(r[k], 3) for k in DIMS}, combo)
