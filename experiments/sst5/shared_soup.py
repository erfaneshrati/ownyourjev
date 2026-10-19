"""Shared-init soup: continue fine-tuning one trained model for E epochs with K data-order seeds, average weights.
usage: shared_soup.py BASE OUT K EPOCHS LR"""
import sys, json; sys.path.insert(0,"experiments/sst5")
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import TrainConfig, _train_gradient, predict_probs
from ownyourjev.data import Example
from lab import *
base, out, K, E, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
L = load("pool_v1_deepseek-v4.1-flash_9000"); excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
exs = [Example(e.uid, e.state, {"answer": int(np.argmax(L[e.uid]["soft"]))}, None, e.group, {"soft": {"answer": L[e.uid]["soft"]}})
       for e in P.pool(T) if e.uid not in excl and L.get(e.uid, {}).get("soft")]
idx = np.random.default_rng(0).permutation(len(exs)); val = [exs[i] for i in idx[:400]]; tr = [exs[i] for i in idx[400:]]
from ownyourjev.gpu import gpu_slot
import contextlib; _slot = gpu_slot(8.0); dev_ = _slot.__enter__()
tok = AutoTokenizer.from_pretrained(f"data/v2/sst5/models/{base}/backbone")
sds = []
for k in range(K):
    m = DecisionModel.load(f"data/v2/sst5/models/{base}", device=dev_)
    cfg = TrainConfig(backbone="x", method="full", lr=lr, epochs=E, batch_size=32, max_len=m.max_len, soft_labels=True, seed=100 + k, patience=10, device=dev_, calibrate=False, warmup_frac=0.0)
    res = _train_gradient(m, tok, Q["answer"], tr, val, cfg, "answer", verbose=False)
    pd = predict_probs(res.model, tok, DEV); print("member", k, "dev", (pd.argmax(1) == DEVY).mean(), flush=True)
    sds.append({kk: v.detach().float().cpu().clone() for kk, v in res.model.state_dict().items()}); del m, res; torch.cuda.empty_cache()
m = DecisionModel.load(f"data/v2/sst5/models/{base}", device="cpu")
ref = m.state_dict()
m.load_state_dict({k: (sum(s[k] for s in sds) / K).to(v.dtype) if v.is_floating_point() else v for k, v in ref.items()})
m.temperature.fill_(1.0); m.to(dev_)
pd = predict_probs(m, tok, DEV); report(out, pd)
d = Path("data/v2/sst5/runs") / out; d.mkdir(parents=True, exist_ok=True); np.save(d / "dev_probs.npy", pd)
m.to("cpu"); m.save(f"data/v2/sst5/models/{out}", tok)
