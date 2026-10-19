"""Train the shared-encoder two-head student on soft LLM labels; report dev metrics per epoch.
Usage: uv run python experiments/toxigen/train_mt.py --name NAME [--labels pool_flash_g1] [--silver 0] [--epochs 3] ..."""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, QS, RES, TASK, best_threshold, load, soft_of  # noqa: E402
from mt_model import MultiHead  # noqa: E402
from benchmarks.v2 import protocol as P  # noqa: E402
from ownyourjev.models.backbones import resolve_backbone  # noqa: E402
from ownyourjev.gpu import gpu_slot  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--labels", default="pool_flash_g1")
ap.add_argument("--silver", type=float, default=0.0, help="weight of v1 silver one-hot labels mixed into targets")
ap.add_argument("--backbone", default="gte-modernbert")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max_len", type=int, default=64)
ap.add_argument("--w_tox", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_train", type=int, default=0)
ap.add_argument("--sharpen", type=float, default=1.0, help="temperature on soft dist targets (<1 sharpens)")
ap.add_argument("--save", action="store_true")
ap.add_argument("--gpu_gb", type=float, default=6.0)
ap.add_argument("--transductive", default="", help="comma list of label-file prefixes (e.g. dev_flash_g1,dev_b_flash_b20) whose texts+ensemble soft labels are added to training (simulates labeling eval_inputs)")
ap.add_argument("--trans_weight", type=int, default=1, help="repeat transductive rows this many times")
ap.add_argument("--mse", type=float, default=0.0, help="aux loss weight: MSE between predicted and target expected toxicity (/4)")
ap.add_argument("--extra_labels", default="", help="comma list of other pool label files to average in")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = P.dev(TASK)
excl = {e.uid for e in dev} | {e.uid for e in P.seeds(TASK)}
pool = P.pool(TASK)
recs = load(DATA / "labels" / f"{a.labels}.jsonl")
extra = [load(DATA / "labels" / f"{n}.jsonl") for n in a.extra_labels.split(",") if n]
silver = {}
if a.silver > 0:
    for f in ["silver", "silver_seeded"]:
        for l in open(P.RESULTS.parents[1] / "data" / "silver" / TASK / f"{f}.jsonl"):
            r = json.loads(l)
            if r["uid"] not in excl:
                silver.setdefault(r["uid"], []).append(soft_of(r))

rows = []
for e in pool:
    if e.uid in excl or e.uid not in recs:
        continue
    s = soft_of(recs[e.uid])
    if s is None:
        continue
    srcs = [s] + [soft_of(x[e.uid]) for x in extra if e.uid in x and soft_of(x[e.uid]) is not None]
    pt = float(np.mean([x[0] for x in srcs])); d = np.mean([x[1] for x in srcs], 0)
    if e.uid in silver:
        sp = np.mean([x[0] for x in silver[e.uid]]); sd = np.mean([x[1] for x in silver[e.uid]], 0)
        pt = (pt + a.silver * sp) / (1 + a.silver); d = (d + a.silver * sd) / (1 + a.silver)
    if a.sharpen != 1.0:
        d = d ** (1 / a.sharpen); d = d / d.sum()
    rows.append((e.state["text"], pt, d.astype(np.float32)))
if a.transductive:
    T = [load(DATA / "labels" / f"{n}.jsonl") for n in a.transductive.split(",")]
    uids = set.intersection(*[set(t) for t in T]); n_t = 0
    for uid in sorted(uids):
        ss = [soft_of(t[uid]) for t in T]; ss = [x for x in ss if x is not None]
        if not ss:
            continue
        txt = T[0][uid]["state"]["text"]
        for _ in range(a.trans_weight):
            rows.append((txt, float(np.mean([x[0] for x in ss])), np.mean([x[1] for x in ss], 0).astype(np.float32))); n_t += 1
    print("transductive rows", n_t)
random.shuffle(rows)
if a.n_train:
    rows = rows[: a.n_train]
print(f"train rows {len(rows)}", flush=True)
dev_texts = [e.state["text"] for e in dev]
y = np.array([bool(e.gold["toxic"]) for e in dev]); g = np.array([float(e.gold["toxicity"]) for e in dev])


def collate(tok):
    def f(b):
        t, pt, d = zip(*b)
        enc = tok(list(t), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(pt), torch.tensor(np.stack(d))
    return f


@torch.no_grad()
def predict(m, tok, texts, dev_):
    m.eval(); out = []
    for i in range(0, len(texts), 128):
        enc = tok(texts[i:i + 128], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m(enc["input_ids"].to(dev_), enc["attention_mask"].to(dev_)).float().cpu())
    return torch.cat(out)


def cv_thr_acc(s):
    """2-fold threshold cross-validation on dev (tune on one half, score the other)."""
    idx = np.arange(len(s)); A, B = idx % 2 == 0, idx % 2 == 1
    tA, _ = best_threshold(s[A], y[A]); tB, _ = best_threshold(s[B], y[B])
    return float((((s[B] >= tA) == y[B]).sum() + ((s[A] >= tB) == y[A]).sum()) / len(s))


def dev_metrics(probs):
    pt = probs[:, 0]; es = probs[:, 1:] @ np.arange(5); comb = 0.5 * pt + 0.5 * es / 4
    r = {"spearman": float(stats.spearmanr(g, es).statistic), "acc_p@0.5": float(((pt >= 0.5) == y).mean()),
         "auroc_p": float(roc_auc_score(y, pt)), "auroc_es": float(roc_auc_score(y, es)), "auroc_comb": float(roc_auc_score(y, comb))}
    for k, s in [("p", pt), ("es", es / 4), ("comb", comb)]:
        t, acc = best_threshold(s, y); r[f"acc_{k}_best"] = acc; r[f"thr_{k}"] = t; r[f"acc_{k}_cv"] = cv_thr_acc(s)
    return r


with gpu_slot(a.gpu_gb) as device:
    torch.cuda.set_device(device)
    spec = resolve_backbone(a.backbone)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    bb = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float32)
    m = MultiHead(bb, bb.config.hidden_size, spec.pooling, 0.1, a.max_len, spec.name).to(device)
    dl = DataLoader(rows, batch_size=a.bs, shuffle=True, collate_fn=collate(tok))
    head_ids = {id(p) for p in m.head.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in m.parameters() if id(p) not in head_ids], "lr": a.lr},
                             {"params": list(m.head.parameters()), "lr": a.lr * 5}], weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    total = len(dl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    hist = []; t0 = time.time(); best = None
    for ep in range(a.epochs):
        m.train()
        for ids, mask, pt, d in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(ids.to(device), mask.to(device))
            loss = MultiHead.loss(z, pt.to(device), d.to(device), a.w_tox)
            if a.mse > 0:
                ar = torch.arange(5, device=device, dtype=torch.float32) / 4
                loss = loss + a.mse * ((torch.softmax(z.float()[:, 1:], -1) @ ar - d.to(device).float() @ ar) ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        probs = m.probs(predict(m, tok, dev_texts, device), calibrated=False).numpy()
        r = {"epoch": ep + 1, **dev_metrics(probs), "elapsed": round(time.time() - t0)}
        hist.append(r); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    out = {"name": a.name, "args": vars(a), "n_train": len(rows), "history": hist}
    with (RES / "dev_train_log.jsonl").open("a") as f:
        f.write(json.dumps(out) + "\n")
    if a.save:
        d = DATA / "models" / a.name
        m.cpu().save(d, tok, QS["toxicity"].model_dump())
        np.save(d / "dev_probs_torch.npy", probs)
        print("saved", d)
