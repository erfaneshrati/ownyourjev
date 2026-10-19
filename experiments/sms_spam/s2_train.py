"""Cross-fit training for SMS spam. usage: s2_train.py <cfg_name> <fold|all> [--labels file]"""
import argparse, random
from common import *
from ownyourjev.train import TrainConfig
T = "sms_spam"
CFGS = {
    "gte_full_e3": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=3, batch_size=32, max_len=128),
    "gte_full_e5": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=5, batch_size=32, max_len=128, patience=2),
    "gte_full_e3_lr5": dict(backbone="gte-modernbert", method="full", lr=5e-5, epochs=3, batch_size=32, max_len=128),
}
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("fold"); ap.add_argument("--labels", default="labels.jsonl"); ap.add_argument("--tag", default="")
a = ap.parse_args()
lab = {e.uid: e for e in load_jsonl(ROOT / "data/v2/sms_spam" / a.labels)}
ev = protocol.eval_inputs(T); folds = protocol.eval_folds(T)
dev = protocol.dev(T)
name = a.cfg + a.tag
outd = ROOT / "data/v2/sms_spam/models" / name; outd.mkdir(parents=True, exist_ok=True)
fl = range(len(folds)) if a.fold == "all" else [int(a.fold)]
for k in fl:
    tr_idx, te_idx = folds[k]
    tr = [lab[ev[i].uid] for i in tr_idx if ev[i].uid in lab]
    random.Random(k).shuffle(tr)
    nv = len(tr) // 10
    va, tr = tr[:nv], tr[nv:]
    cfg = TrainConfig(**CFGS[a.cfg], seed=k, gpu_need_gb=6.0)
    res = train_save(T, tr, va, cfg, outd / f"fold{k}")
    pdev = torch_probs(res, dev)
    pte = torch_probs(res, [ev[i] for i in te_idx])
    np.save(outd / f"fold{k}/dev_probs.npy", pdev); np.save(outd / f"fold{k}/te_probs.npy", pte); np.save(outd / f"fold{k}/te_idx.npy", np.array(te_idx))
    y = [e.labels["answer"] for e in dev]
    print(f"[{name} fold{k}] best_epoch={res.best_epoch} dev acc/f1@0.5 = {acc_f1(y, pdev)}  best-thr {best_thr(y, pdev)}", flush=True)
    del res
