"""Custom student training (own loop, library unchanged): gte-modernbert full FT on soft 6-level targets
(average of labelers' two-hot/soft distributions), optional expected-score MSE term, optional ordinal smoothing;
epoch chosen by Spearman on a held-out SILVER val split (never dev). Dev Spearman reported per epoch."""
import argparse, json, math, random, time, os
from dataclasses import replace
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from common import *
from ownyourjev.train import TrainConfig, _build_model, _Collate, predict_logits
from ownyourjev.spec import render_state
from ownyourjev.gpu import gpu_slot

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--weights", default="")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init_seed", type=int, default=-1, help="seed for model/head init (default = --seed); data order/dropout use --seed")
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--backbone", default="gte-modernbert")
ap.add_argument("--gpu_gb", type=float, default=4.0)
ap.add_argument("--mse", type=float, default=0.0, help="weight of MSE between expected level and target expectation")
ap.add_argument("--sigma", type=float, default=0.0, help="blur targets with a Gaussian over levels (ordinal smoothing)")
ap.add_argument("--hard", action="store_true", help="round averaged expectation to one-hot")
ap.add_argument("--add_gold", default="", help="'seeds' or 'seeds+dev0'/'seeds+dev1' (train on other dev half, score on fold)")
ap.add_argument("--n", type=int, default=0, help="subsample pool")
ap.add_argument("--full", action="store_true", help="no silver val split: train on all, use fixed --epochs")
ap.add_argument("--tag", required=True)
ap.add_argument("--extra", default="", help="comma list of extra splits (new_gen,new_repair) labeled by the same labelers")
ap.add_argument("--extra_n", type=int, default=0)
ap.add_argument("--extra_w", type=float, default=1.0)
ap.add_argument("--swap", action="store_true", help="augment: add each pair with sentences swapped")
ap.add_argument("--save", action="store_true")
ap.add_argument("--eval", action="store_true")
a = ap.parse_args()
K = 6; LV = np.arange(K)

def onehot(i):
    v = np.zeros(K); v[int(i)] = 1; return v
def twohot(x):
    x = min(max(float(x), 0.0), K - 1.0); v = np.zeros(K); lo = int(np.floor(x)); hi = min(lo + 1, K - 1); w = x - lo
    v[lo] += 1 - w; v[hi] += w; return v

excl = {e.uid for e in protocol.seeds(T)} | {e.uid for e in protocol.dev(T)}
pool = {e.uid: e for e in protocol.pool(T)}
files = a.labels.split(","); ws = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(files)
acc = {}
for f, w in zip(files, ws):
    if f in ("v1", "v1s"):
        for l in open(ROOT / f"data/silver/stsb/{'silver' if f == 'v1' else 'silver_seeded'}.jsonl"):
            r = json.loads(l)
            if r["uid"] in pool and r["labels"]:
                acc.setdefault(r["uid"], []).append((w, onehot(r["labels"]["answer"])))
    else:
        for e in load_jsonl(D / f"labels_pool_{f}.jsonl"):
            if e.uid in pool and e.labels:
                s = (e.meta or {}).get("soft", {}).get("answer")
                acc.setdefault(e.uid, []).append((w, np.array(s) if s is not None else onehot(e.labels["answer"])))
need = len(files)
items = []
for uid, vs in acc.items():
    if uid in excl or len(vs) < need: continue
    p = sum(w * v for w, v in vs) / sum(w for w, _ in vs)
    st = pool[uid].state
    items.append((render_state(st), p, render_state({"sentence_1": st["sentence_2"], "sentence_2": st["sentence_1"]})))
rng = random.Random(0); rng.shuffle(items)
if a.n: items = items[:a.n]
if a.full:
    val = items[:300]; tr = items  # val only for monitoring
else:
    nval = 300; val, tr = items[:nval], items[nval:]
dev = protocol.dev(T); gdev = np.array([e.gold["answer"] for e in dev]); score_idx = np.arange(len(dev))
for sp_ in [x for x in a.extra.split(",") if x]:
    st_ = {r["uid"]: r["state"] for r in map(json.loads, open(D / f"{sp_}.jsonl"))}
    acc2 = {}
    for f in files:
        for e in load_jsonl(D / f"labels_{sp_}_{f}.jsonl"):
            if e.labels:
                acc2.setdefault(e.uid, []).append(np.array(e.meta["soft"]["answer"]))
    add = []
    for uid in st_:
        vs = acc2.get(uid, [])
        if len(vs) < need: continue
        st = st_[uid]
        add.append((render_state(st), np.mean(vs, 0), render_state({"sentence_1": st["sentence_2"], "sentence_2": st["sentence_1"]})))
    if a.extra_n: add = add[:a.extra_n]
    print("extra", sp_, len(add), flush=True)
    tr = tr + add
if a.add_gold:
    gold = list(protocol.seeds(T))
    if "dev" in a.add_gold:
        k = int(a.add_gold[-1]); gold += [e for i, e in enumerate(dev) if i % 2 != k]; score_idx = np.array([i for i in range(len(dev)) if i % 2 == k])
    tr = tr + [(render_state(e.state), twohot(e.gold["answer"])) for e in gold]

def blur(p):
    if a.sigma <= 0: return p
    M = np.exp(-((LV[:, None] - LV[None, :]) ** 2) / (2 * a.sigma ** 2)); M /= M.sum(1, keepdims=True)
    q2 = p @ M; return q2 / q2.sum()
if a.swap:
    tr = tr + [(x[2], x[1]) for x in tr if len(x) == 3]
tr_t = []
for t, p, *_ in tr:
    if a.hard: p = onehot(int(round(float(p @ LV))))
    tr_t.append((t, blur(p).tolist()))
print(f"n_train={len(tr_t)} n_val={len(val)}", flush=True)

def train_once():
  with gpu_slot(a.gpu_gb) as dev_name:
        torch.cuda.set_device(dev_name)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        cfg = TrainConfig(backbone=a.backbone, method="full", lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=a.max_len, seed=a.seed, device=dev_name)
        if a.init_seed >= 0: torch.manual_seed(a.init_seed)
        model, tok = _build_model(q, cfg)
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        g = torch.Generator(); g.manual_seed(a.seed)
        dl = DataLoader(tr_t, batch_size=a.bs, shuffle=True, collate_fn=_Collate(tok, a.max_len), generator=g)
        head = set(id(p) for p in model.head.parameters())
        dec = [p for n, p in model.named_parameters() if p.ndim > 1 and id(p) not in head]
        nod = [p for n, p in model.named_parameters() if p.ndim <= 1 and id(p) not in head]
        opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01, "lr": a.lr}, {"params": nod, "weight_decay": 0.0, "lr": a.lr},
                                 {"params": list(model.head.parameters()), "weight_decay": 0.0, "lr": a.lr * 5}], betas=(0.9, 0.98), eps=1e-6)
        total = len(dl) * a.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
        lv = torch.arange(K, dtype=torch.float32, device=dev_name)
        vtexts = [x[0] for x in val]; vexp = np.array([x[1] @ LV for x in val])
        dtexts = [render_state(e.state) for e in dev]
        best = (-1, None, 0); hist = []
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            for ids, mask, y in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids.to(dev_name), mask.to(dev_name))
                y = y.to(dev_name); lp = F.log_softmax(logits.float(), -1)
                loss = -(y * lp).sum(-1).mean()
                if a.mse > 0:
                    loss = loss + a.mse * ((lp.exp() @ lv - y @ lv) ** 2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            lvv = predict_logits(model, tok, vtexts, max_len=a.max_len, device=dev_name)
            ld = predict_logits(model, tok, dtexts, max_len=a.max_len, device=dev_name)
            ev = torch.softmax(torch.as_tensor(lvv), -1).numpy() @ LV; ed = torch.softmax(torch.as_tensor(ld), -1).numpy() @ LV
            vs, ds_ = sp(vexp, ev), sp(gdev[score_idx], ed[score_idx])
            hist.append((ep + 1, round(vs, 4), round(ds_, 4))); print("epoch", hist[-1], f"{time.time()-t0:.0f}s", flush=True)
            if vs > best[0] or a.full:
                best = (vs, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
        model.load_state_dict(best[1]); model.eval()
        ld = predict_logits(model, tok, dtexts, max_len=a.max_len, device=dev_name)
        pd = torch.softmax(torch.as_tensor(ld), -1).numpy(); ed = pd @ LV
        ld64 = predict_logits(model, tok, dtexts, max_len=64, device=dev_name); dev64 = sp(gdev, torch.softmax(torch.as_tensor(ld64), -1).numpy() @ LV)
        print("dev at max_len 64:", round(dev64, 4), flush=True)
        dsw = [render_state({"sentence_1": e.state["sentence_2"], "sentence_2": e.state["sentence_1"]}) for e in dev]
        pd_t = (pd + torch.softmax(torch.as_tensor(predict_logits(model, tok, dsw, max_len=a.max_len, device=dev_name)), -1).numpy()) / 2
        h = np.arange(len(dev)) % 2 == 0
        out = {"tag": a.tag, "args": vars(a), "dev_spearman": sp(gdev[score_idx], ed[score_idx]), "dev_A": sp(gdev[h], ed[h]), "dev_B": sp(gdev[~h], ed[~h]),
               "dev_pearson": float(np.corrcoef(gdev[score_idx], ed[score_idx])[0, 1]), "dev_spearman_tta": sp(gdev[score_idx], (pd_t @ LV)[score_idx]), "best_epoch": best[2], "hist": hist, "n_train": len(tr_t)}
        print("RESULT", json.dumps(out), flush=True)
        od = D / "runs" / a.tag; od.mkdir(parents=True, exist_ok=True)
        np.save(od / "dev_probs.npy", pd); np.save(od / "dev_probs_tta.npy", pd_t); (od / "result.json").write_text(json.dumps(out, indent=1))
        with open(D / "runs.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
        if a.save:
            model.save(D / "models" / a.tag, tok)
        if a.eval:
            le = predict_logits(model, tok, [render_state(e.state) for e in protocol.eval_inputs(T)], max_len=a.max_len, device=dev_name)
            np.save(od / "eval_probs.npy", torch.softmax(torch.as_tensor(le), -1).numpy())
            esw = [render_state({"sentence_1": e.state["sentence_2"], "sentence_2": e.state["sentence_1"]}) for e in protocol.eval_inputs(T)]
            le2 = predict_logits(model, tok, esw, max_len=a.max_len, device=dev_name)
            np.save(od / "eval_probs_tta.npy", (torch.softmax(torch.as_tensor(le), -1).numpy() + torch.softmax(torch.as_tensor(le2), -1).numpy()) / 2)

for attempt in range(30):
    try:
        train_once(); break
    except torch.OutOfMemoryError:
        pass
    import gc; gc.collect(); torch.cuda.empty_cache(); print("OOM, retrying in 60s", flush=True); time.sleep(60)
