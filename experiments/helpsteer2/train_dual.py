"""Shared encoder with labeler-specific heads: 5 heads fit deepseek-flash labels, 5 heads fit qwen labels (each loss
masked to examples that labeler covered). Deployment combines heads (chosen on gold dev). Epoch selected on gold dev."""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from common import *
import mh

ap = argparse.ArgumentParser()
ap.add_argument("--flash", nargs="*", default=["silver_v1.jsonl"])
ap.add_argument("--qwen", nargs="*", default=["qwen_silver.jsonl", "qwen_restall.jsonl"])
ap.add_argument("--layers", type=int, default=22)
ap.add_argument("--budget", type=int, default=128)
ap.add_argument("--p_frac", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--sigma", type=float, default=0.6)
ap.add_argument("--wq", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--silver_val", default="sval1000.jsonl")
ap.add_argument("--no_sval", action="store_true")
ap.add_argument("--gold", default=None, help="'i/k': add gold dev examples not in fold i (of k) as training data for BOTH head sets; evaluate on fold i only. 'all': train on all dev+seeds (no dev eval)")
ap.add_argument("--gold_w", type=float, default=5.0)
ap.add_argument("--fixed_epoch", type=int, default=0, help="if >0, keep the model after this epoch (no dev selection)")
ap.add_argument("--rank_w", type=float, default=0.0, help="pairwise logistic ranking loss on same-prompt pairs (flash heads, flash/gold targets)")
ap.add_argument("--rank_scale", type=float, default=2.0)
ap.add_argument("--hc_override", nargs="*", default=[], help="jsonl files whose helpfulness/correctness labels replace the flash h/c targets (stronger labeler)")
ap.add_argument("--hc_only_override", action="store_true", help="train flash heads only on rows that have an h/c override (probe)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
idx5 = np.arange(5)

def gauss(x):
    g = np.exp(-((idx5 - x) ** 2) / (2 * args.sigma ** 2)); return g / g.sum()

hold = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
sval = [] if args.no_sval else load_jsonl(LD / args.silver_val)
sval_u = {e.uid for e in sval}
data = {}
for src, key in [(args.flash, "f"), (args.qwen, "q")]:
    for f in src:
        for e in load_jsonl(LD / f):
            if not e.labels or e.uid in hold or e.uid in sval_u: continue
            dec = (e.meta or {}).get("dec")
            t = np.stack([gauss(float(dec[a]) if dec else float(e.labels[a])) for a in A])
            d = data.setdefault(e.uid, {"state": e.state, "f": None, "q": None, "w": 1.0}); d[key] = t
hc_ov = {}
for f in args.hc_override:
    for e in load_jsonl(LD / f):
        if e.labels and e.uid not in hold and e.uid not in sval_u:
            hc_ov[e.uid] = e.labels
for u, d in data.items():
    d["prompt"] = d["state"]["prompt"]
    if u in hc_ov and d["f"] is not None:
        d["f"] = d["f"].copy()
        for j, a in enumerate(A[:2]):
            if a in hc_ov[u]: d["f"][j] = gauss(float(hc_ov[u][a]))
        d["ov"] = True
if args.hc_only_override:
    for u, d in data.items():
        if not d.get("ov"): d["f"] = None
print("h/c overrides applied:", sum(1 for d in data.values() if d.get("ov")), flush=True)
rows = list(data.values())
dev = protocol.dev(T)
if args.gold:
    gold_pool = protocol.dev(T) + protocol.seeds(T)
    if args.gold == "all":
        tr_gold, dev = gold_pool, protocol.dev(T)
    else:
        i, k = map(int, args.gold.split("/"))
        tr_gold = [e for j, e in enumerate(protocol.dev(T)) if j % k != i] + protocol.seeds(T)
        dev = [e for j, e in enumerate(protocol.dev(T)) if j % k == i]
    for e in tr_gold:
        t = np.stack([gauss(float(e.gold[a])) for a in A])
        rows.append({"state": e.state, "f": t, "q": t, "w": args.gold_w, "prompt": e.state["prompt"]})
print(f"train {len(rows)} (flash {sum(r['f'] is not None for r in rows)}, qwen {sum(r['q'] is not None for r in rows)}) sval {len(sval)} dev {len(dev)}", flush=True)

from ownyourjev.gpu import gpu_slot
with gpu_slot(float(os.environ.get("NEED_GB", "4"))) as dev_str:
    torch.cuda.set_device(dev_str)
    model, tok = mh.build(args.layers)
    model.heads = torch.nn.Linear(model.backbone.config.hidden_size, 10 * mh.K); model.n_heads = 10
    model.register_buffer("temperature", torch.ones(10))
    enc = mh.Encoder(tok, budget=args.budget, p_frac=args.p_frac)
    model.to(dev_str)
    for r in rows: r["ids"] = enc.ids(r["state"])
    dev_ids = [enc.ids(e.state) for e in dev]; sval_ids = [enc.ids(e.state) for e in sval]
    Z = np.zeros((5, 5))

    def collate(b):
        L = max(len(r["ids"]) for r in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long); m = torch.zeros(len(b), L, dtype=torch.long)
        for i, r in enumerate(b):
            ids[i, :len(r["ids"])] = torch.tensor(r["ids"]); m[i, :len(r["ids"])] = 1
        return ids.to(dev_str), m.to(dev_str)

    @torch.no_grad()
    def predict(ids_list):
        model.eval(); order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i])); res = [None] * len(ids_list)
        for s in range(0, len(order), 64):
            ix = order[s:s + 64]; ids, m = collate([{"ids": ids_list[i]} for i in ix])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.probs(model(ids, m), calibrated=False).float().cpu().numpy()
            for i, pi in zip(ix, p): res[i] = pi
        return np.stack(res)

    def combos(P):
        es = P @ np.arange(5); f, q = es[:, :5], es[:, 5:]
        zf = (f - f.mean(0)) / (f.std(0) + 1e-9); zq = (q - q.mean(0)) / (q.std(0) + 1e-9)
        return {"f": f, "q": q, "avg": zf + zq}

    def score(E, gold):
        rs = [sp(E[:, j], gold[:, j]) for j in range(5)]; return rs + [float(np.mean(rs))]

    G = np.array([[e.gold[a] for a in A] for e in dev], float)
    SV = np.array([[e.meta["x"][a] for a in A] for e in sval], float) if sval else None
    head_ids = {id(p) for p in model.heads.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
                             {"params": list(model.heads.parameters()), "lr": args.lr * 5}], weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    nb = math.ceil(len(rows) / args.bs); total = nb * args.epochs
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    best, hist = (-9, None, None, None), []
    for ep in range(args.epochs):
        model.train(); tl = 0; random.shuffle(rows)
        if args.rank_w > 0:  # keep same-prompt rows adjacent so pairs share a batch
            grp = {}
            for r in rows: grp.setdefault(r["prompt"], []).append(r)
            gl = list(grp.values()); random.shuffle(gl); flat = [r for g in gl for r in g]
            batches = [flat[i:i + args.bs] for i in range(0, len(flat), args.bs)]; random.shuffle(batches)
        else:
            batches = [rows[i:i + args.bs] for i in range(0, len(rows), args.bs)]
        for b in batches:
            ids, m = collate(b)
            yf = torch.tensor(np.stack([r["f"] if r["f"] is not None else Z for r in b]), dtype=torch.float32, device=dev_str)
            yq = torch.tensor(np.stack([r["q"] if r["q"] is not None else Z for r in b]), dtype=torch.float32, device=dev_str)
            mf = torch.tensor([float(r["f"] is not None) * r["w"] for r in b], device=dev_str)
            mq = torch.tensor([float(r["q"] is not None) * r["w"] for r in b], device=dev_str)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(ids, m)
            lp = F.log_softmax(lg.float(), -1)
            lf = -(yf * lp[:, :5]).sum(-1).mean(-1); lq = -(yq * lp[:, 5:]).sum(-1).mean(-1)
            loss = (lf * mf).sum() / mf.sum().clamp(min=1) + args.wq * (lq * mq).sum() / mq.sum().clamp(min=1)
            if args.rank_w > 0:
                pr = [r["prompt"] for r in b]
                pairs = [(i, j) for i in range(len(b)) for j in range(i + 1, len(b)) if pr[i] == pr[j] and b[i]["f"] is not None and b[j]["f"] is not None]
                if pairs:
                    ar = torch.arange(5, device=dev_str, dtype=torch.float32)
                    es = (lp[:, :5].exp() * ar).sum(-1); ys = (yf * ar).sum(-1)
                    ii = torch.tensor([p[0] for p in pairs], device=dev_str); jj = torch.tensor([p[1] for p in pairs], device=dev_str)
                    dlt = ys[ii] - ys[jj]; msk = (dlt.abs() > 0.25).float()
                    rl = (F.softplus(-torch.sign(dlt) * (es[ii] - es[jj]) * args.rank_scale) * msk).sum() / msk.sum().clamp(min=1)
                    loss = loss + args.rank_w * rl
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tl += float(loss)
        P = predict(dev_ids); C = combos(P)
        rec = {"epoch": ep + 1, "train_loss": tl / nb, "dev": {k: score(v, G) for k, v in C.items()}}
        if sval:
            Cs = combos(predict(sval_ids)); rec["sval"] = {k: score(v, SV) for k, v in Cs.items()}
        hist.append(rec)
        print(f"ep{ep+1} loss {tl/nb:.4f} " + " | ".join(f"dev-{k} " + " ".join(f"{x:.3f}" for x in v) for k, v in rec["dev"].items())
              + ("  || sval-f %.3f sval-q %.3f" % (rec["sval"]["f"][-1], rec["sval"]["q"][-1]) if sval else ""), flush=True)
        key = max(rec["dev"]["f"][-1], rec["dev"]["avg"][-1])
        if (args.fixed_epoch and ep + 1 == args.fixed_epoch) or (not args.fixed_epoch and key > best[0]):
            best = (key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, P, ep + 1)
    model.load_state_dict(best[1])
    out = Path(args.out); mh.save(model.cpu(), tok, enc, out, {"args": vars(args), "history": hist, "best_epoch": best[3], "heads_layout": "0-4 flash, 5-9 qwen"})
    np.save(out / "dev_probs10.npy", best[2])
    print("BEST", best[3], f"{best[0]:.4f}", flush=True)
