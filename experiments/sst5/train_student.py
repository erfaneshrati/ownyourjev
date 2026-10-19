"""Train gte-modernbert student on (soft) LLM labels over the pool; report dev metrics; save dev/eval probs.
usage: train_student.py OUTNAME --labels a,b[,..] [--weights ...] [--hard] [--epochs N] [--lr X] [--gold-seeds]"""
import argparse, json, sys
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, "experiments/sst5")
from lab import load, report, DEV, DEVY, SEEDS, Q, T, P
from ownyourjev.train import train_with_oom_retry, TrainConfig, predict_probs
from ownyourjev.data import Example

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--labels", required=True); ap.add_argument("--weights", default=None)
ap.add_argument("--hard", action="store_true"); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sharpen", type=float, default=1.0)
ap.add_argument("--backbone", default="gte-modernbert"); ap.add_argument("--eval", action="store_true")
ap.add_argument("--save", action="store_true"); ap.add_argument("--gold-fold", default=None); ap.add_argument("--extra", default=None); ap.add_argument("--pooling", default=None); ap.add_argument("--gold-dup", type=int, default=3); ap.add_argument("--max-len", type=int, default=128)
a = ap.parse_args()

q = Q["answer"]
srcs = [load(n) for n in a.labels.split(",")]
w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(srcs)
dev_uids = {e.uid for e in DEV} | {e.uid for e in SEEDS}
pool = [e for e in P.pool(T) if e.uid not in dev_uids]
exs = []
for e in pool:
    ps, ws = [], []
    for s, wi in zip(srcs, w):
        r = s.get(e.uid)
        if r is None or r["labels"] is None: continue
        if r.get("soft"): v = np.array(r["soft"], float)
        else: v = np.zeros(5); v[r["labels"]["answer"]] = 1
        ps.append(v); ws.append(wi)
    if not ps: continue
    p = np.average(ps, axis=0, weights=ws)
    if a.sharpen != 1.0: p = p ** a.sharpen
    p = p / p.sum()
    lab = int(p.argmax())
    exs.append(Example(e.uid, e.state, {"answer": lab}, None, e.group, {"soft": {"answer": p.tolist()}}))
rng = np.random.default_rng(0); idx = rng.permutation(len(exs))
val = [exs[i] for i in idx[:400]]; tr = [exs[i] for i in idx[400:]]
if a.extra:  # extra transfer texts (e.g. phrases) with their own label file; train-only
    XL = load(a.extra); n0 = len(tr)
    for r in map(json.loads, open(f"data/v2/sst5/labels/{a.extra}_texts.jsonl")):
        l = XL.get(r["uid"])
        if l and l["labels"] and l.get("soft"):
            p = np.array(l["soft"]); tr.append(Example(r["uid"], r["state"], {"answer": int(p.argmax())}, None, "", {"soft": {"answer": p.tolist()}}))
    print("extra added", len(tr) - n0, flush=True)
held = np.ones(len(DEV), bool)
if a.gold_fold is not None:  # add gold (seeds + dev outside fold k of K) to training; report only on fold k
    k, K = map(int, a.gold_fold.split("/"))
    fold = np.random.default_rng(123).permutation(len(DEV)) % K
    held = fold == k if K > 1 else np.zeros(len(DEV), bool)
    gold = list(SEEDS) + [e for e, h in zip(DEV, held) if not h]
    tr += [Example(e.uid, e.state, dict(e.labels), None, e.group, {}) for e in gold] * a.gold_dup
    print("gold added", len(gold), "x", a.gold_dup, "held", held.sum(), flush=True)
print("n_train", len(tr), "label dist", np.bincount([x.labels["answer"] for x in tr], minlength=5), flush=True)
cfg = TrainConfig(backbone=a.backbone, method="full", lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=a.max_len,
                  soft_labels=not a.hard, seed=a.seed, pooling=a.pooling, patience=10, gpu_need_gb=float(__import__("os").environ.get("GPUGB","6")))
res = train_with_oom_retry(q, tr, val, cfg)
print("history", json.dumps(res.history), flush=True)
out = Path("data/v2/sst5/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
pd = predict_probs(res.model, res.tokenizer, DEV)
np.save(out / "dev_probs.npy", pd)
np.save(out / "dev_held.npy", held)
acc = report(a.name, pd[held], DEVY[held]) if held.any() else float("nan")
if a.eval:
    np.save(out / "eval_probs.npy", predict_probs(res.model, res.tokenizer, P.eval_inputs(T)))
if a.save:
    res.model.save(Path("data/v2/sst5/models") / a.name, res.tokenizer)
json.dump({"args": vars(a), "dev_acc": acc, "best_epoch": res.best_epoch, "temp": res.temperature}, open(out / "summary.json", "w"))
