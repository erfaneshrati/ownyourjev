"""Train a gte-modernbert student on LLM labels, evaluate on dev gold (never trained on dev).

usage: train_student.py TASK NAME --data file1.jsonl[,file2] [--n N] [--epochs E] [--max-len L] [--soft] [--lr LR]
Data files are jsonl of Examples (labels / meta.soft); 'v1:silver' or 'v1:silver_seeded' loads v1 labels.
Validation for early stopping + temperature = 10% held-out *silver* (not dev). Writes
data/v2/TASK/models/NAME/ (model), dev_probs.npy, and appends a row to data/v2/TASK/student_runs.jsonl.
"""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("name")
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--soft", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--add-seeds", action="store_true", help="add the 20 gold seeds to training")
ap.add_argument("--min-conf", type=float, default=0.0, help="drop examples whose soft max-prob < this")
ap.add_argument("--balance", action="store_true", help="oversample minority classes to the median class count")
ap.add_argument("--eval", action="store_true", help="also predict eval_inputs (for the frozen recipe)")
ap.add_argument("--backbone", default="gte-modernbert")
a = ap.parse_args()

from ownyourjev.train import TrainConfig, train_with_oom_retry, predict_probs
from ownyourjev.evaluate import evaluate
from ownyourjev.data import train_val_split
from dataclasses import replace

T = a.task
D, R = paths(T)
q = protocol.task_spec(T).parsed["answer"]
held = {e.uid for e in protocol.seeds(T)} | {e.uid for e in protocol.dev(T)}
exs = {}
for src in a.data.split(","):
    if src.startswith("v1:"):
        part = v1_silver(T, src[3:])
    else:
        part = {e.uid: e for e in load_jsonl(D / src if not src.startswith("/") else src)}
    for u, e in part.items():
        if u in held or not e.labels or e.labels.get("answer") is None:
            continue
        exs.setdefault(u, e)  # first source wins
data = sorted(exs.values(), key=lambda e: e.uid)
if a.min_conf:
    data = [e for e in data if max((e.meta.get("soft") or {}).get("answer", [1.0])) >= a.min_conf]
random.Random(a.seed).shuffle(data)
if a.n:
    data = data[: a.n]
tr, va = train_val_split(data, 0.1, seed=0)
if a.balance:
    from collections import Counter
    cnt = Counter(e.labels["answer"] for e in tr); med = int(np.median(list(cnt.values())))
    rng = random.Random(1); extra = []
    for k, c in cnt.items():
        if c < med:
            pool_k = [e for e in tr if e.labels["answer"] == k]
            extra += [rng.choice(pool_k) for _ in range(med - c)]
    tr = tr + extra
if a.add_seeds:
    tr = tr + protocol.seeds(T)
cfg = TrainConfig(backbone=a.backbone, method="full", lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=a.max_len,
                  soft_labels=a.soft, seed=a.seed, patience=2, gpu_need_gb=6.0)
t0 = time.time()
res = train_with_oom_retry(q, tr, va, cfg)
dev = protocol.dev(T)
p = predict_probs(res.model, res.tokenizer, dev)
golds = [e.labels["answer"] for e in dev]
m = evaluate(q, golds, p)
pa = evaluate(q, golds[0::2], p[0::2])["accuracy"]; pb = evaluate(q, golds[1::2], p[1::2])["accuracy"]
out = D / "models" / a.name
res.model.save(out, tokenizer=res.tokenizer)
np.save(out / "dev_probs.npy", p)
row = {"name": a.name, "args": vars(a), "n_train": len(tr), "val_silver_acc": res.val_metrics.get("accuracy"),
       "best_epoch": res.best_epoch, "dev_acc": m["accuracy"], "dev_A": pa, "dev_B": pb, "dev_macro_f1": m.get("macro_f1"),
       "train_s": round(time.time() - t0)}
if a.eval:
    pe = predict_probs(res.model, res.tokenizer, protocol.eval_inputs(T))
    np.save(out / "eval_probs.npy", pe)
    row["eval_saved"] = True
with open(D / "student_runs.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("RESULT", json.dumps(row))
