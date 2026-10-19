"""Uniform weight averaging ("model soup") of several fine-tuned students from the same init; evaluates on dev."""
import sys, numpy as np, torch
from common import *
from mh import MH, predict_logits, texts_of
out_name, names = sys.argv[1], sys.argv[2:]
ms = [MH.load(D / "models" / n) for n in names]
m, tok = ms[0]
sd = {k: sum(x[0].state_dict()[k].float() for x in ms) / len(ms) for k in m.state_dict()}
m.load_state_dict(sd); m.temperature.fill_(1.0)
dev = protocol.dev(T)
dvc = "cpu"
m = m.to(dvc)
pd = torch.sigmoid(torch.as_tensor(predict_logits(m, tok, texts_of(dev), dvc))).numpy()
o = D / "runs" / out_name; o.mkdir(parents=True, exist_ok=True); np.save(o / "dev_probs.npy", pd)
for i, h in enumerate(HEADS):
    y = [e.gold[h] for e in dev]
    print(report(y, pd[:, i], f"{out_name}:{h}"), f"cvF1={cv_threshold_f1(y, pd[:, i]):.3f}")
# also the probability-average ensemble of the members, for reference
pe = np.mean([np.load(D / "runs" / n / "dev_probs.npy") for n in names], 0)
for i, h in enumerate(HEADS):
    y = [e.gold[h] for e in dev]
    print(report(y, pe[:, i], f"probavg:{h}"), f"cvF1={cv_threshold_f1(y, pe[:, i]):.3f}")
m.cpu().save(D / "models" / out_name, tok)
