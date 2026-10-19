import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.v2 import protocol
from ownyourjev.data import Example
T = "helpsteer2"
A = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
D = ROOT / "data/v2/helpsteer2"; D.mkdir(parents=True, exist_ok=True)
LD = D / "labels"; LD.mkdir(exist_ok=True)
R = ROOT / "benchmarks/results_v2/helpsteer2"
Q = protocol.task_spec(T).parsed

def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(stats.spearmanr(a[m], b[m]).statistic)

def save_jsonl(path, exs):
    with open(path, "w") as f:
        for e in exs:
            f.write(json.dumps({"uid": e.uid, "state": e.state, "labels": e.labels, "group": e.group, "meta": e.meta}) + "\n")

def load_jsonl(path):
    out = []
    for l in open(path):
        r = json.loads(l)
        out.append(Example(r["uid"], r["state"], r["labels"], None, r.get("group", ""), r.get("meta", {})))
    return out

def expected(e, a):
    s = (e.meta or {}).get("soft", {}).get(a)
    if s is not None:
        return float(np.dot(s, np.arange(len(s))))
    return float(e.labels[a]) if e.labels else np.nan

def report(name, labeled, gold_exs):
    g = {e.uid: e.gold for e in gold_exs}
    rows = {}
    for part, sl in [("A", slice(0, 100)), ("B", slice(100, 200)), ("all", slice(0, 200))]:
        sub = gold_exs[sl]; lab = {e.uid: e for e in labeled}
        rs = [sp([expected(lab[x.uid], a) for x in sub], [g[x.uid][a] for x in sub]) for a in A]
        rows[part] = rs + [float(np.mean(rs))]
    print(f"{name:40s} " + " | ".join(f"{p}: " + " ".join(f"{v:.3f}" for v in r) for p, r in rows.items()), flush=True)
    return rows
