"""Train gte-modernbert student on pool labels; report dev Spearman (dev never used for training/selection inside train)."""
import argparse, json, random
from dataclasses import replace
from common import *
from ownyourjev.train import TrainConfig, train_with_oom_retry, predict_probs

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="comma list of label files (pool) to average as soft targets; 'v1'/'v1s' = v1 silver")
ap.add_argument("--mode", default="soft", choices=["soft", "hard", "ordinal"])
ap.add_argument("--sigma", type=float, default=0.0)
ap.add_argument("--sharpen", type=float, default=1.0, help="temperature on soft targets (<1 sharper)")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--backbone", default="gte-modernbert")
ap.add_argument("--add_gold", action="store_true", help="add seeds (+ dev fold) as soft two-hot targets")
ap.add_argument("--dev_fold", type=int, default=-1, help="with add_gold: train on dev except fold k of 2 (score on fold k)")
ap.add_argument("--tag", required=True)
ap.add_argument("--save", action="store_true")
ap.add_argument("--eval", action="store_true", help="also write eval_inputs probs")
a = ap.parse_args()
K = q.num_outputs

def onehot(i):
    v = np.zeros(K); v[int(i)] = 1; return v

def twohot(x):
    v = np.zeros(K); lo = int(np.floor(x)); hi = min(lo + 1, K - 1); w = x - lo
    v[lo] += 1 - w; v[hi] += w; return v

excl = {e.uid for e in protocol.seeds(T)} | {e.uid for e in protocol.dev(T)}
pool = {e.uid: e for e in protocol.pool(T)}
acc = {}
for f in a.labels.split(","):
    if f in ("v1", "v1s"):
        path = ROOT / f"data/silver/stsb/{'silver' if f == 'v1' else 'silver_seeded'}.jsonl"
        for l in open(path):
            r = json.loads(l)
            if r["uid"] in pool and r["labels"]:
                acc.setdefault(r["uid"], []).append(onehot(r["labels"]["answer"]))
    else:
        for e in load_jsonl(D / f):
            if e.uid in pool and e.labels:
                s = (e.meta or {}).get("soft", {}).get("answer")
                acc.setdefault(e.uid, []).append(np.array(s) if s is not None else onehot(e.labels["answer"]))
train = []
for uid, vs in acc.items():
    if uid in excl: continue
    p = np.mean(vs, 0)
    if a.sharpen != 1.0:
        p = p ** (1 / a.sharpen); p = p / p.sum()
    exp = float(p @ np.arange(K))
    if a.mode == "soft":
        tgt = p
    elif a.mode == "ordinal":
        tgt = None
    train.append(Example(uid, pool[uid].state, {"answer": int(round(exp)) if a.mode != "hard" else int(p.argmax())}, None, "", {"soft": {"answer": list(map(float, tgt))}} if a.mode == "soft" else {}))
rng = random.Random(0); rng.shuffle(train)
nval = max(200, len(train) // 20)
val, train = train[:nval], train[nval:]
dev = protocol.dev(T)
score_idx = list(range(len(dev)))
if a.add_gold:
    gold = list(protocol.seeds(T))
    if a.dev_fold >= 0:
        gold += [e for i, e in enumerate(dev) if i % 2 != a.dev_fold]
        score_idx = [i for i in range(len(dev)) if i % 2 == a.dev_fold]
    for e in gold:
        x = e.gold["answer"]
        if a.mode == "soft":
            train.append(Example(e.uid, e.state, {"answer": int(round(x))}, None, "", {"soft": {"answer": list(map(float, twohot(x)))}}))
        else:
            train.append(Example(e.uid, e.state, {"answer": int(round(x))}, None, "", {}))
print(f"n_train={len(train)} n_val={len(val)}", flush=True)
cfg = TrainConfig(backbone=a.backbone, method="full", lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=a.max_len, seed=a.seed,
                  soft_labels=(a.mode == "soft"), ordinal_sigma=a.sigma if a.mode == "ordinal" else 0.0, patience=2, gpu_need_gb=5.0)
res = train_with_oom_retry(q, train, val, cfg)
print("history", [(h["epoch"], round(h["val_loss"], 4), round(h.get("val_spearman", 0), 4)) for h in res.history], "best", res.best_epoch, flush=True)
pd = predict_probs(res.model, res.tokenizer, dev)
g = np.array([e.gold["answer"] for e in dev]); s = pd @ np.arange(K)
si = np.array(score_idx)
h = np.arange(len(dev)) % 2 == 0
out = {"tag": a.tag, "args": vars(a), "dev_spearman": sp(g[si], s[si]), "dev_A": sp(g[h], s[h]), "dev_B": sp(g[~h], s[~h]), "best_epoch": res.best_epoch, "n_train": len(train)}
print("RESULT", json.dumps(out), flush=True)
od = D / "runs" / a.tag; od.mkdir(parents=True, exist_ok=True)
np.save(od / "dev_probs.npy", pd)
(od / "result.json").write_text(json.dumps(out, indent=1))
with open(D / "runs.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
if a.save:
    md = D / "models" / a.tag
    res.model.save(md, res.tokenizer)
if a.eval:
    pe = predict_probs(res.model, res.tokenizer, protocol.eval_inputs(T))
    np.save(od / "eval_probs.npy", pe)
