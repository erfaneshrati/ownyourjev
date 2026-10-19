"""5-fold CV over dev: train on LLM-labeled pool + synth + seeds (+ the 4 other dev folds x dup), predict the held-out dev fold.
usage: i4_cv.py <cfg> <dup>   (dup=0: dev not used in training, same folds, for a paired comparison)"""
import sys, random, gc, torch
from common import *
from ownyourjev.data import kfold
from ownyourjev.train import TrainConfig
import i3_train_lib as L
T = "prompt_injections"; cfg_name = sys.argv[1]; dup = int(sys.argv[2])
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
base = L.data("pool_synth_seeds"); random.Random(0).shuffle(base)
nv = len(base) // 10; va, base = base[:nv], base[nv:]
folds = kfold(dev, 5, seed=0)
oof = np.zeros(len(dev)); name = f"cv_{cfg_name}_dup{dup}"
for k, (tr_i, te_i) in enumerate(folds):
    tr = base + [dev[i] for i in tr_i] * dup
    res = train_save(T, tr, va, TrainConfig(**L.CFGS[cfg_name], seed=k, gpu_need_gb=6.0), L.D / "models" / name / f"fold{k}")
    oof[te_i] = torch_probs(res, [dev[i] for i in te_i])
    res.model.cpu(); del res; gc.collect(); torch.cuda.empty_cache()
np.save(L.D / f"oof_{name}.npy", oof)
print(f"[{name}] oof acc@0.5 {acc_f1(y, oof)[0]:.3f} best-thr {best_thr(y, oof, 'acc')} acc@best {acc_f1(y, oof, best_thr(y, oof, 'acc'))[0]:.3f}", flush=True)
