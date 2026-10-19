"""PI final: (1) int8 check on the 5 dup3 CV fold models (oof int8 acc); (2) train final gte_e8 on pool+synth+seeds+dev x3, export int8, eval probs."""
import random, gc, torch
from common import *
from ownyourjev.data import kfold
from ownyourjev.train import TrainConfig
import i3_train_lib as L
T = "prompt_injections"
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev]); ev = protocol.eval_inputs(T)
folds = kfold(dev, 5, seed=0); oof = np.zeros(len(dev))
for k, (tr_i, te_i) in enumerate(folds):
    d = L.D / "models/cv_gte_e8_dup3" / f"fold{k}"
    if not (d / "model.int8.onnx").is_file(): export_int8(d)
    oof[te_i] = onnx_probs(d, [dev[i] for i in te_i])
np.save(L.D / "oof_cv_gte_e8_dup3_int8.npy", oof)
print("[cv int8] oof acc", [(t, round(acc_f1(y, oof, t)[0], 3)) for t in [0.1, 0.3, 0.5]], flush=True)
base = L.data("pool_synth_seeds"); random.Random(0).shuffle(base)
nv = len(base) // 10; va, base = base[:nv], base[nv:]
outd = L.D / "models/final_gte_e8_dup3"
res = train_save(T, base + dev * 3, va, TrainConfig(**L.CFGS["gte_e8"], seed=0, gpu_need_gb=6.0), outd)
pt = torch_probs(res, ev); res.model.cpu(); del res; gc.collect(); torch.cuda.empty_cache()
export_int8(outd)
pq = onnx_probs(outd, ev)
np.save(L.D / "eval_probs.npy", pq.astype(np.float32))
print("[final] eval torch/int8 agreement @0.5", float(((pt >= .5) == (pq >= .5)).mean()), "pos rate int8", float((pq >= .5).mean()), flush=True)
