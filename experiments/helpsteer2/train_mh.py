"""Train the shared-encoder 5-head scorer on LLM labels; select the epoch on gold dev (mean Spearman)."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from common import *
import mh

ap = argparse.ArgumentParser()
ap.add_argument("--labels", nargs="+", required=True, help="jsonl files (labels and optional meta.soft); later files override earlier by uid")
ap.add_argument("--layers", type=int, default=22)
ap.add_argument("--budget", type=int, default=128)
ap.add_argument("--p_frac", type=float, default=0.3)
ap.add_argument("--no_header", action="store_true")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--sigma", type=float, default=0.6, help="ordinal smoothing of hard labels")
ap.add_argument("--soft", action="store_true", help="use meta.soft when present")
ap.add_argument("--pooling", default="cls")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rank_w", type=float, default=0.0, help="weight of within-prompt pairwise ranking loss on expected scores")
ap.add_argument("--extra_train", nargs="*", default=[], help="extra jsonl (e.g. gold folds / synthetic) added to training")
ap.add_argument("--extra_w", type=float, default=1.0)
ap.add_argument("--dev_slice", default="all", help="'all' or 'i/k' -> evaluate on dev fold i of k (others may be in extra_train)")
ap.add_argument("--out", required=True)
ap.add_argument("--save_preds", action="store_true")
ap.add_argument("--silver_val", default=None, help="jsonl of held-out LLM-labelled examples (excluded from training); report Spearman vs their labels")
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

hold = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
by = {}
for f in args.labels:
    for e in load_jsonl(f):
        if e.labels and e.uid not in hold:
            by[e.uid] = e
sval = load_jsonl(args.silver_val) if args.silver_val else []
sval_u = {e.uid for e in sval}
train = [e for u, e in by.items() if u not in sval_u]
extra = []
for f in args.extra_train:
    extra += [e for e in load_jsonl(f) if e.labels]
dev = protocol.dev(T)
if args.dev_slice != "all":
    i, k = map(int, args.dev_slice.split("/"))
    dev = [e for j, e in enumerate(dev) if j % k == i]
print(f"train {len(train)} extra {len(extra)} dev {len(dev)}", flush=True)

def target(e):
    out = []
    for a in A:
        s = (e.meta or {}).get("soft", {}).get(a) if args.soft else None
        if s is None:
            y = int(e.labels[a]); idx = np.arange(mh.K)
            s = np.exp(-((idx - y) ** 2) / (2 * args.sigma ** 2)) if args.sigma > 0 else (idx == y).astype(float)
            s = s / s.sum()
        out.append(np.asarray(s, float))
    return np.stack(out)

from ownyourjev.gpu import gpu_slot
import os
with gpu_slot(float(os.environ.get("NEED_GB", "3"))) as dev_str:
    torch.cuda.set_device(dev_str)
    model, tok = mh.build(args.layers, args.pooling)
    enc = mh.Encoder(tok, budget=args.budget, p_frac=args.p_frac, header=not args.no_header)
    model.to(dev_str)
    t0 = time.time()
    rows = [(enc.ids(e.state), target(e), 1.0, e.state["prompt"]) for e in train] + [(enc.ids(e.state), target(e), args.extra_w, e.state["prompt"]) for e in extra]
    dev_ids = [enc.ids(e.state) for e in dev]
    sval_ids = [enc.ids(e.state) for e in sval]
    print(f"tokenised in {time.time()-t0:.0f}s", flush=True)

    def batches(rows, bs, shuffle):
        if args.rank_w > 0 and shuffle:  # keep same-prompt pairs in the same batch
            groups = {}
            for r in rows: groups.setdefault(r[3], []).append(r)
            gl = list(groups.values()); random.shuffle(gl)
            flat = [r for g in gl for r in g]
            out, cur = [], []
            for r in flat:
                cur.append(r)
                if len(cur) >= bs: out.append(cur); cur = []
            if cur: out.append(cur)
            random.shuffle(out); return out
        idx = list(range(len(rows)))
        if shuffle: random.shuffle(idx)
        return [[rows[i] for i in idx[s:s + bs]] for s in range(0, len(idx), bs)]

    def collate(b):
        L = max(len(r[0]) for r in b)
        ids = torch.zeros(len(b), L, dtype=torch.long) + (tok.pad_token_id or 0); m = torch.zeros(len(b), L, dtype=torch.long)
        for i, r in enumerate(b):
            ids[i, :len(r[0])] = torch.tensor(r[0]); m[i, :len(r[0])] = 1
        return ids.to(dev_str), m.to(dev_str)

    @torch.no_grad()
    def predict(ids_list):
        model.eval(); out = []
        order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
        res = [None] * len(ids_list)
        for s in range(0, len(order), 64):
            ix = order[s:s + 64]
            ids, m = collate([(ids_list[i],) for i in ix])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.probs(model(ids, m), calibrated=False).float().cpu().numpy()
            for i, pi in zip(ix, p): res[i] = pi
        return np.stack(res)

    def dev_eval(P):
        es = P @ np.arange(mh.K)
        rs = [sp(es[:, j], [e.gold[a] for e in dev]) for j, a in enumerate(A)]
        return rs + [float(np.mean(rs))]

    params = [p for p in model.parameters()]
    head_ids = {id(p) for p in model.heads.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in head_ids], "lr": args.lr},
                             {"params": list(model.heads.parameters()), "lr": args.lr * 5}], weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    nb = math.ceil(len(rows) / args.bs); total = nb * args.epochs
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    best, hist = (-9, None, None), []
    ar = torch.arange(mh.K, device=dev_str, dtype=torch.float32)
    for ep in range(args.epochs):
        model.train(); tl = 0
        for b in batches(rows, args.bs, True):
            ids, m = collate(b)
            y = torch.tensor(np.stack([r[1] for r in b]), dtype=torch.float32, device=dev_str)
            w = torch.tensor([r[2] for r in b], dtype=torch.float32, device=dev_str)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(ids, m)
            lp = F.log_softmax(lg.float(), -1)
            loss = (-(y * lp).sum(-1).mean(-1) * w).sum() / w.sum()
            if args.rank_w > 0:
                es = (lp.exp() * ar).sum(-1)  # (B, H)
                ys = (y * ar).sum(-1)
                pr = [r[3] for r in b]
                ii, jj = zip(*[(i, j) for i in range(len(b)) for j in range(len(b)) if i < j and pr[i] == pr[j]]) if any(pr[i] == pr[j] for i in range(len(b)) for j in range(i + 1, len(b))) else ((), ())
                if ii:
                    ii, jj = torch.tensor(ii, device=dev_str), torch.tensor(jj, device=dev_str)
                    d = ys[ii] - ys[jj]; sgn = torch.sign(d); mask = (d.abs() > 0.25).float()
                    rl = (F.softplus(-sgn * (es[ii] - es[jj]) * 2.0) * mask).sum() / mask.sum().clamp(min=1)
                    loss = loss + args.rank_w * rl
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tl += float(loss)
        P = predict(dev_ids); r = dev_eval(P)
        sv = None
        if sval:
            es = predict(sval_ids) @ np.arange(mh.K)
            sv = [sp(es[:, j], [e.meta["x"][a] if "x" in (e.meta or {}) else e.labels[a] for e in sval]) for j, a in enumerate(A)]
            sv.append(float(np.mean(sv)))
        hist.append({"epoch": ep + 1, "train_loss": tl / nb, "dev": r, "silver_val": sv})
        print(f"ep{ep+1} loss {tl/nb:.4f} dev " + " ".join(f"{v:.3f}" for v in r) + (" | sval " + " ".join(f"{v:.3f}" for v in sv) if sv else ""), flush=True)
        if r[-1] > best[0]:
            best = (r[-1], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, P)
    model.load_state_dict(best[1])
    out = Path(args.out); mh.save(model.cpu(), tok, enc, out, {"args": vars(args), "history": hist, "best_dev_mean": best[0]})
    if args.save_preds:
        np.save(out / "dev_probs.npy", best[2])
    print("BEST", f"{best[0]:.4f}", json.dumps(hist[-1]), flush=True)
