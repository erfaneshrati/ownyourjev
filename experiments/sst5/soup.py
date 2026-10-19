"""Uniform weight average ('model soup') of fine-tuned DecisionModels sharing a backbone init. usage: soup.py OUT in1 in2 ..."""
import sys; sys.path.insert(0,"experiments/sst5")
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from lab import DEV, DEVY, report
out, ins = sys.argv[1], sys.argv[2:]
ms = [DecisionModel.load(f"data/v2/sst5/models/{n}", device="cpu") for n in ins]
sd = {k: (sum(m.state_dict()[k].float() for m in ms) / len(ms)).to(v.dtype) if v.is_floating_point() else v for k, v in ms[0].state_dict().items()}
m = ms[0]; m.load_state_dict(sd); tok = AutoTokenizer.from_pretrained(f"data/v2/sst5/models/{ins[0]}/backbone")
m.temperature.fill_(1.0)
m.to("cuda:1" if torch.cuda.is_available() else "cpu")
pd = predict_probs(m, tok, DEV); report(out, pd)
m.to("cpu"); m.save(f"data/v2/sst5/models/{out}", tok)
d = Path("data/v2/sst5/runs") / out; d.mkdir(parents=True, exist_ok=True); np.save(d / "dev_probs.npy", pd)
