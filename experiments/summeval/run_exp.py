"""Train one multi-head student and evaluate on dev gold (dev articles are never in training here).
usage: run_exp.py <name> --train lab_eval__g1__qwen3.7-flash[,lab_synth_x...] [--holdout 0] [cfg overrides k=v ...]
holdout=k excludes eval fold k's articles (real and synthetic) from training; -1 = use all eval folds."""
import argparse, json, sys, time
import numpy as np, torch
from common import *
from ownyourjev.data import read_jsonl
import mh

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--train", required=True); ap.add_argument("--holdout", type=int, default=0)
ap.add_argument("--dev_in_train", action="store_true"); ap.add_argument("--save", default="")
ap.add_argument("--predict_eval", action="store_true"); ap.add_argument("--dev_half", type=int, default=-1); ap.add_argument("--dev_llm", default=""); ap.add_argument("--verify", default="")
ap.add_argument("kv", nargs="*")
a = ap.parse_args()
cfg = mh.Cfg()
for kv in a.kv:
    k, v = kv.split("="); t = type(getattr(cfg, k)); setattr(cfg, k, (v == "True") if t is bool else t(v))

ev = P.eval_inputs(T); folds = P.eval_folds(T)
fold_of = {}
for k, (_, te) in enumerate(folds):
    for i in te: fold_of[ev[i].group] = k
dev_groups = {e.group for e in P.dev(T) + P.seeds(T)}

train = []
for src in a.train.split(","):
    for e in read_jsonl(DATA / f"{src}.jsonl"):
        if e.labels is None: continue
        g = e.group
        if g in dev_groups and not a.dev_in_train: continue
        if a.holdout >= 0 and fold_of.get(g, -9) == a.holdout: continue
        train.append(e)
if a.dev_in_train:
    train += P.dev(T) + P.seeds(T)
if a.verify:
    from dataclasses import replace as _r
    VV = {}
    for f in a.verify.split(","):
        for e in read_jsonl(DATA / f"{f}.jsonl"):
            if e.meta.get("verify"): VV.setdefault(e.uid, []).append(e.meta["verify"]["v_mean"])
    train = [_r(e, meta={**e.meta, "verify": {"v_mean": float(np.mean(VV[e.uid]))}}) if e.uid in VV else e for e in train]
    print("verify targets", sum(e.uid in VV for e in train), "/", len(train))
dev = P.dev(T)
if a.dev_half >= 0:
    dg = sorted({e.group for e in dev}); half = set(dg[a.dev_half::2])
    if a.dev_llm:
        from dataclasses import replace as _rep
        train += [_rep(e, gold=None) for e in read_jsonl(DATA / f"{a.dev_llm}.jsonl") if e.group in half and e.labels]
    else:
        train += [e for e in P.dev(T) + P.seeds(T) if e.group in half]
    dev = [e for e in dev if e.group not in half]
devX = mh.featurize(dev)
print(a.name, "n_train", len(train), cfg, flush=True)
dev_h = DATA / "dev_hold"; dev_h.mkdir(exist_ok=True)

def eval_fn(model, tok):
    p = mh.predict(model, tok, dev, device, feats=devX)
    pred = {d: p[:, j] @ np.arange(5) for j, d in enumerate(cfg.dims)}
    r = score_dev(pred, dev)
    if model.aux is not None:
        g = [e.group for e in dev]; ax = model.last_aux
        r["aux_vs_cons"] = group_spearman([e.labels["consistency"] for e in dev], ax, g)
        for lam in (0.5, 1.0, 2.0):
            r[f"cons_mix{lam}"] = group_spearman([e.labels["consistency"] for e in dev], pred["consistency"] + lam * 4 * ax, g)
    return {k: round(v, 4) for k, v in r.items()}

from ownyourjev.gpu import gpu_slot
for attempt in range(20):
    try:
        with gpu_slot(float(__import__("os").environ.get("NEED_GB", "4"))) as device:
            torch.cuda.set_device(device)
            model, tok, hist = mh.train(cfg, train, device, eval_fn=eval_fn)
            p = mh.predict(model, tok, dev, device, feats=devX); dev_aux = model.last_aux
            pe = None
            if a.predict_eval:
                te = folds[a.holdout][1] if a.holdout >= 0 else list(range(len(ev)))
                pe = mh.predict(model, tok, [ev[i] for i in te], device)
        break
    except torch.OutOfMemoryError:
        print("OOM, retrying", flush=True)
        import gc; model = None; gc.collect(); torch.cuda.empty_cache(); time.sleep(60)
if False:
    p = mh.predict(model, tok, dev, device, feats=devX)
    pe = None
    if a.predict_eval:
        te = folds[a.holdout][1] if a.holdout >= 0 else list(range(len(ev)))
        pe = mh.predict(model, tok, [ev[i] for i in te], device)
rec = {"name": a.name, "train": a.train, "holdout": a.holdout, "cfg": cfg.__dict__, "hist": hist, "n_train": len(train)}
with open(RES / "dev_student_log.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
np.save(dev_h / f"{a.name}.npy", p)
if getattr(model, "last_aux", None) is not None and not np.isnan(model.last_aux).all():
    np.save(dev_h / f"{a.name}.aux.npy", dev_aux)
if pe is not None:
    (DATA / "eval_preds").mkdir(exist_ok=True); np.save(DATA / "eval_preds" / f"{a.name}.npy", pe)
if a.save:
    mh.save(model, tok, DATA / "models" / a.save)
print("FINAL", a.name, hist[-1])
