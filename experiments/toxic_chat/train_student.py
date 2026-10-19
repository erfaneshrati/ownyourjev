"""Train the shared-encoder two-head student on pool labels; report dev metrics; save dev/eval probs.
usage: train_student.py NAME LABELS.jsonl [--epochs 4] [--pw 1,1] [--hard] [--seed 0] [--extra extra.jsonl] [--eval]"""
import argparse, json, random
import numpy as np, torch
from common import *
from mh import train_mh, predict_logits, texts_of

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("labels", nargs="+")
ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--pw", default="1,1"); ap.add_argument("--hard", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--extra", nargs="+", default=None, help="extra (synthetic) examples jsonl, train only")
ap.add_argument("--eval", action="store_true", help="also predict eval_inputs and save model")
ap.add_argument("--save", action="store_true", help="save model dir")
ap.add_argument("--device", default="cuda:1"); ap.add_argument("--oversample", type=int, default=1)
a = ap.parse_args()

block = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
pool_uids = {e.uid for e in protocol.pool(T)}
# merge label files: average soft probs across files (ensemble), by uid
by = {}
for f in a.labels:
    for e in load_jsonl(f):
        if e.uid in block or e.uid not in pool_uids or e.labels is None:
            continue
        by.setdefault(e.uid, [e, []])[1].append([soft(e, h) if not a.hard else float(bool(e.labels[h])) for h in HEADS])
items = [(render, np.mean(ys, 0).tolist()) for (e, ys) in by.values() for render in [texts_of([e])[0]]]
print("train items", len(items), "mean pos", np.mean([y for _, y in items], 0))
rng = random.Random(0); rng.shuffle(items)
nval = max(200, len(items) // 10)
val, train = items[:nval], items[nval:]
if a.oversample > 1:
    train = train + [it for it in train if it[1][0] >= 0.5] * (a.oversample - 1)
if a.extra:
    ex = [e for f in a.extra for e in load_jsonl(f)]
    train += [(t, [soft(e, h) for h in HEADS]) for e, t in zip(ex, texts_of(ex))]
print("train", len(train), "val", len(val))
pw = tuple(float(x) for x in a.pw.split(","))
from ownyourjev.gpu import gpu_slot
import time, gc
for attempt in range(8):
    try:
        with gpu_slot(4.0) as _dev:
            a.device = _dev
            m, tok, ep = train_mh(train, val, epochs=a.epochs, lr=a.lr, pos_weight=pw, seed=a.seed, max_len=a.max_len, device=a.device, bs=8, accum=2)
        break
    except torch.OutOfMemoryError:
        print("OOM, retrying", attempt, flush=True); gc.collect(); torch.cuda.empty_cache(); time.sleep(90)
with gpu_slot(3.0) as _dev2:
    a.device = _dev2; m = m.to(_dev2)
    dev = protocol.dev(T)
    pd = torch.sigmoid(torch.as_tensor(predict_logits(m, tok, texts_of(dev), a.device))).numpy()
    out = D / "runs" / a.name; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dev_probs.npy", pd)
    res = {"name": a.name, "args": vars(a), "best_epoch": ep}
    for i, h in enumerate(HEADS):
        y = [e.gold[h] for e in dev]
        r = report(y, pd[:, i], f"{a.name}:{h}") + f" cvF1={cv_threshold_f1(y, pd[:, i]):.3f}"
        print(r); res[h] = r
    if a.eval or a.save:
        m.save(D / "models" / a.name, tok)
    if a.eval:
        ev = protocol.eval_inputs(T)
        pe = torch.sigmoid(torch.as_tensor(predict_logits(m, tok, texts_of(ev), a.device))).numpy()
        np.save(out / "eval_probs.npy", pe)
    (out / "result.json").write_text(json.dumps(res, indent=1))
