"""Teacher pseudo-labels for the unlabeled PAWS pool (texts without v1 LLM labels). usage: p5_pseudo.py <teacher_model_name>"""
import sys, torch
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs, pick_device
from transformers import AutoTokenizer
T = "paws"
d = ROOT / "data/v2/paws/models" / sys.argv[1]
have = set(load_v1(T, "silver_seeded"))
rest = [e for e in protocol.pool(T) if e.uid not in have]
dev_ = pick_device(6.0)
m = DecisionModel.load(d, device=dev_); tok = AutoTokenizer.from_pretrained(d / "backbone")
p = predict_probs(m, tok, rest, batch_size=128).reshape(-1)
out = [Example(e.uid, e.state, {"answer": bool(pi >= 0.5)}, None, "", {"soft": {"answer": float(pi)}, "src": "pseudo:" + sys.argv[1]}) for e, pi in zip(rest, p)]
save_jsonl(ROOT / f"data/v2/paws/pseudo_{sys.argv[1]}.jsonl", out)
print("pseudo", len(out), "pos", int((p >= 0.5).sum()), "confident(>0.9|<0.1)", int(((p > 0.9) | (p < 0.1)).sum()))
