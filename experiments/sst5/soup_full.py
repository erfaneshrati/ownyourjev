"""Soup N fine-tuned models -> refit head on silver soft labels (no gold) -> save. usage: soup_full.py OUT LABELS(comma) in1 in2 ..."""
import sys; sys.path.insert(0,"experiments/sst5")
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import embed_texts, _texts, predict_probs
from lab import *
out, labs, ins = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
ms = [DecisionModel.load(f"data/v2/sst5/models/{n}", device="cpu") for n in ins]
sd = {k: (sum(m.state_dict()[k].float() for m in ms) / len(ms)).to(v.dtype) if v.is_floating_point() else v for k, v in ms[0].state_dict().items()}
m = ms[0]; m.load_state_dict(sd); del ms
tok = AutoTokenizer.from_pretrained(f"data/v2/sst5/models/{ins[0]}/backbone")
dev_ = "cuda:1" if torch.cuda.is_available() else "cpu"; m.to(dev_)
Ls = [load(l) for l in labs]; excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
pool = [e for e in P.pool(T) if e.uid not in excl and all(L.get(e.uid, {}).get("soft") for L in Ls)]
S = np.array([np.mean([L[e.uid]["soft"] for L in Ls], 0) for e in pool])
X = embed_texts(m, tok, _texts(pool), max_len=128, use_cache=False)
Xr = np.repeat(X, 5, 0); yr = np.tile(np.arange(5), len(X)); wr = S.reshape(-1); keep = wr > 0.01
clf = LogisticRegression(C=0.1, max_iter=3000).fit(Xr[keep], yr[keep], sample_weight=wr[keep])
with torch.no_grad():
    m.head.weight.copy_(torch.as_tensor(clf.coef_, dtype=m.head.weight.dtype)); m.head.bias.copy_(torch.as_tensor(clf.intercept_, dtype=m.head.bias.dtype))
m.temperature.fill_(1.0)
pd = predict_probs(m, tok, DEV); report(out, pd)
d = Path("data/v2/sst5/runs") / out; d.mkdir(parents=True, exist_ok=True); np.save(d / "dev_probs.npy", pd)
m.to("cpu"); m.save(f"data/v2/sst5/models/{out}", tok)
