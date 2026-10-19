"""Uniform weight soup of fine-tuned students (same backbone init) -> one model (no extra latency)."""
import sys, shutil, torch
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
out = Path(sys.argv[1]); srcs = sys.argv[2:]
ms = [DecisionModel.load(D / "models" / s) for s in srcs]
sd = {k: sum(m.state_dict()[k].float() for m in ms) / len(ms) for k in ms[0].state_dict()}
m = ms[0]; m.load_state_dict(sd); m.temperature.fill_(1.0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(D / "models" / srcs[0] / "backbone")
m.save(out, tok)
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
p = predict_probs(m, tok, dev, device="cpu")
print("soup", srcs, "dev spearman", round(sp(g, p @ np.arange(6)), 4))
np.save(out / "dev_probs_torch.npy", p)
