import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.v2 import protocol
from ownyourjev.data import Example
T = "toxic_chat"
D = ROOT / "data/v2/toxic_chat"; D.mkdir(parents=True, exist_ok=True)
R = ROOT / "benchmarks/results_v2/toxic_chat"; R.mkdir(parents=True, exist_ok=True)
Q = protocol.task_spec(T).parsed
HEADS = ["toxic", "jailbreak"]


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


def soft(e, h):
    s = (e.meta or {}).get("soft", {}).get(h)
    if s is not None:
        return float(s)
    return float(bool(e.labels[h])) if e.labels else np.nan


def f1_at(y, p, t=0.5):
    return float(f1_score(np.asarray(y, bool), np.asarray(p) >= t, zero_division=0))


def best_f1(y, p):
    y = np.asarray(y, bool); p = np.asarray(p)
    best = (0, 0.5)
    for t in np.unique(np.round(p, 4)):
        f = f1_at(y, p, t)
        if f > best[0]:
            best = (f, float(t))
    return best


def report(y, p, name=""):
    y = np.asarray(y, bool); p = np.nan_to_num(np.asarray(p, float), nan=0.0)
    bf, bt = best_f1(y, p)
    ap = average_precision_score(y, p) if y.any() else float("nan")
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return f"{name}: F1@.5={f1_at(y, p):.3f} AP={ap:.3f} AUC={auc:.3f} bestF1={bf:.3f}@{bt:.3f} npos_pred={int((p>=.5).sum())}/{int(y.sum())}"


def cv_threshold_f1(y, p, k=5, reps=20, seed=0):
    """Honest estimate of F1 when the threshold is tuned on dev: choose t on k-1 folds, apply on the held-out fold;
    F1 computed on the pooled out-of-fold predictions. Averaged over repeats (stratified)."""
    y = np.asarray(y, bool); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    res = []
    for r in range(reps):
        pos, neg = np.where(y)[0], np.where(~y)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        fold = np.empty(len(y), int)
        fold[pos] = np.arange(len(pos)) % k; fold[neg] = np.arange(len(neg)) % k
        pred = np.zeros(len(y), bool)
        for f in range(k):
            tr = fold != f
            _, t = best_f1(y[tr], p[tr])
            pred[fold == f] = p[fold == f] >= t
        res.append(f1_score(y, pred, zero_division=0))
    return float(np.mean(res))
