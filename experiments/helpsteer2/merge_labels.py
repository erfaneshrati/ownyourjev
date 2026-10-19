"""Merge labelers into one training file: per attribute, x = mean of available labeler scores (flash v1 hard level,
qwen decimal rating); soft target = discretised Gaussian over levels centred at x (sigma). Usage: merge_labels.py out.jsonl sigma src..."""
import sys
from common import *
out, sigma, srcs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
hold = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
scores, states = {}, {}
for s in srcs:
    for e in load_jsonl(LD / s):
        if not e.labels or e.uid in hold: continue
        dec = (e.meta or {}).get("dec")
        v = {a: float(dec[a]) if dec else float(e.labels[a]) for a in A}
        scores.setdefault(e.uid, []).append(v); states[e.uid] = e.state
res = []
idx = np.arange(5)
for u, vs in scores.items():
    x = {a: float(np.clip(np.mean([v[a] for v in vs]), 0, 4)) for a in A}
    soft = {}
    for a in A:
        g = np.exp(-((idx - x[a]) ** 2) / (2 * sigma ** 2)); soft[a] = (g / g.sum()).tolist()
    res.append(Example(u, states[u], {a: int(round(x[a])) for a in A}, None, "", {"soft": soft, "x": x, "n_labelers": len(vs)}))
save_jsonl(LD / out, res)
print(len(res), sum(len(v) > 1 for v in scores.values()))
