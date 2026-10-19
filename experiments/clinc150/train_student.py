"""Train a student and score it on dev + silver val + held-out synthetic oos.
Usage: train_student.py <name> [--synth N] [--oos_up K] [--epochs E] [--lr LR] [--backbone B] [--seeds]"""
import argparse, json, random
from dataclasses import replace
from pathlib import Path
import numpy as np
from ownyourjev.data import read_jsonl
from ownyourjev.train import TrainConfig, train_with_oom_retry, predict_probs
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--synth", type=int, default=0); ap.add_argument("--oos_up", type=int, default=1)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--backbone", default="gte-modernbert"); ap.add_argument("--method", default="full")
ap.add_argument("--seeds", action="store_true"); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--extra", default="", help="extra jsonl of labeled examples to add to train")
ap.add_argument("--full", action="store_true", help="also train on the silver val split (final model)")
ap.add_argument("--ls", type=float, default=0.0); ap.add_argument("--synth2", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

pool = load_pool()
tr, va = split_pool(pool)
syn = read_jsonl("data/v2/clinc150/synth_oos.jsonl") if Path("data/v2/clinc150/synth_oos.jsonl").exists() else []
random.Random(0).shuffle(syn)
syn_va, syn_tr = syn[:300], syn[300:]
train_ex = tr + (va if a.full else [])
oos_tr = [e for e in train_ex if e.labels["answer"] == "oos"]
train_ex += oos_tr * (a.oos_up - 1)
if a.synth:
    train_ex += syn_tr[: a.synth]
syn2 = read_jsonl("data/v2/clinc150/synth_oos2.jsonl") if Path("data/v2/clinc150/synth_oos2.jsonl").exists() else []
random.Random(1).shuffle(syn2)
syn2_va, syn2_tr = syn2[:250], syn2[250:]
if a.synth2:
    train_ex += syn2_tr
if a.seeds:
    train_ex += P.seeds(TASK)
if a.extra:
    train_ex += [e for e in read_jsonl(a.extra) if e.labels]
print("train", len(train_ex), "oos in train", sum(e.labels["answer"] == "oos" for e in train_ex), "val", len(va))
cfg = TrainConfig(backbone=a.backbone, method=a.method, lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=64,
                  patience=2, gpu_need_gb=5.0, label_smoothing=a.ls, seed=a.seed)
res = train_with_oom_retry(Q, train_ex, va, cfg)
print(res.summary())
out = Path("data/v2/clinc150/models") / a.name
res.model.save(out, res.tokenizer)
dev = P.dev(TASK)
pd = predict_probs(res.model, res.tokenizer, dev)
pv = predict_probs(res.model, res.tokenizer, va)
ps = predict_probs(res.model, res.tokenizer, syn_va) if syn_va else None
np.save(out / "dev_probs.npy", pd)
if syn2_va: np.save(out / "syn2va_probs.npy", predict_probs(res.model, res.tokenizer, syn2_va))
np.save(out / "val_probs.npy", pv)
if ps is not None: np.save(out / "synva_probs.npy", ps)
gd = [e.labels["answer"] for e in dev]; gv = [e.labels["answer"] for e in va]
rep = {"args": vars(a), "summary": res.summary(), "thr": {}}
for tau in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    md, mv = metrics(apply_threshold(pd, tau), gd), metrics(apply_threshold(pv, tau), gv)
    ms = float((apply_threshold(ps, tau).argmax(1) == OOS).mean()) if ps is not None else float("nan")
    rep["thr"][tau] = {"dev": md, "val": mv, "synva_oos_recall": ms}
    print(f"tau={tau:.1f} dev acc={md['acc']:.3f} ins={md['ins_acc']:.3f} oos={md['oos_recall']:.2f} | val acc={mv['acc']:.3f} ins={mv['ins_acc']:.3f} oosrec={mv['oos_recall']:.2f} (n={mv['n_oos']}) | synva oos={ms:.2f}")
(out / "report.json").write_text(json.dumps(rep, indent=1))
