import random
import numpy as np
from collections import defaultdict
from ownyourjev.data import read_jsonl
from benchmarks.v2 import protocol as P

TASK = "clinc150"
Q = P.task_spec(TASK).parsed["answer"]
OPTS = Q.options
OOS = OPTS.index("oos")
TEST_OOS_SHARE = 0.18  # public CLINC150 test composition (1000 oos / 5500), given in the brief


def split_pool(exs, val_frac=0.08, seed=0):
    by = defaultdict(list)
    for e in exs:
        by[e.labels["answer"]].append(e)
    rng = random.Random(seed)
    tr, va = [], []
    for k in sorted(by):
        xs = by[k][:]
        rng.shuffle(xs)
        n = max(1, round(len(xs) * val_frac))
        va += xs[:n]; tr += xs[n:]
    return tr, va


def load_pool(path="data/v2/clinc150/pool_labeled.jsonl"):
    return [e for e in read_jsonl(path) if e.labels]


def apply_threshold(p, tau):
    """Predict oos when the best in-scope probability is below tau (raise oos prob above the max)."""
    p = p.copy()
    ins = np.delete(p, OOS, axis=1).max(1)
    m = ins < tau
    p[m, OOS] = ins[m] + 1e-3
    return p / p.sum(1, keepdims=True)


def metrics(p, gold):
    y = np.array([OPTS.index(g) for g in gold])
    pred = p.argmax(1)
    oos = y == OOS
    ins_acc = float((pred[~oos] == y[~oos]).mean()) if (~oos).any() else float("nan")
    oos_rec = float((pred[oos] == OOS).mean()) if oos.any() else float("nan")
    return {"acc": float((pred == y).mean()), "ins_acc": ins_acc, "oos_recall": oos_rec, "n_oos": int(oos.sum()),
            "pred_oos_rate_ins": float((pred[~oos] == OOS).mean()) if (~oos).any() else float("nan"),
            "weighted": (1 - TEST_OOS_SHARE) * ins_acc + TEST_OOS_SHARE * oos_rec}
