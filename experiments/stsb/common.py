import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.v2 import protocol
from ownyourjev.data import Example
T = "stsb"
D = ROOT / "data/v2/stsb"; D.mkdir(parents=True, exist_ok=True)
R = ROOT / "benchmarks/results_v2/stsb"
Q = protocol.task_spec(T).parsed
q = Q["answer"]

def sp(a, b):
    return float(stats.spearmanr(a, b).statistic)

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

def expected(e):
    s = (e.meta or {}).get("soft", {}).get("answer")
    if s is not None:
        return float(np.dot(s, np.arange(len(s))))
    return float(e.labels["answer"]) if e.labels else np.nan
