"""Uniform weight averaging (model soup) of fine-tuned models from the same pretrained backbone."""
import sys, json, shutil, numpy as np, torch
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from transformers import AutoTokenizer
out = Path(sys.argv[1]); dirs = [Path(x) for x in sys.argv[2:]]
ms = [DecisionModel.load(d) for d in dirs]
sd = {k: sum(m.state_dict()[k].float() for m in ms) / len(ms) for k in ms[0].state_dict()}
m = ms[0]; m.load_state_dict(sd); m.temperature.fill_(1.0)
tok = AutoTokenizer.from_pretrained(dirs[0] / "backbone")
dev = protocol.dev(TASK); g = np.array([e.gold[QID] for e in dev])
m = m.to("cuda:0" if torch.cuda.is_available() else "cpu")
p = predict_probs(m, tok, dev, batch_size=32)
from sklearn.metrics import roc_auc_score
print("soup", [d.name for d in dirs], "dev_acc", np.mean((p >= .5) == g), "auroc", round(roc_auc_score(g, p), 3))
m.cpu(); m.save(out, tok); np.save(out / "dev_probs.npy", p)
