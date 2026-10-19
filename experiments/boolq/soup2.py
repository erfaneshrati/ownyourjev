"""Weight soup + Platt recalibration on the runs' shared silver early-stopping split (never dev gold).
usage: soup2.py OUT TRAIN_JSONL MODEL_DIR..."""
import sys, random, numpy as np, torch
from common import *
from trainer import load_set
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_logits, predict_probs, _texts
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
out, train_file, dirs = Path(sys.argv[1]), sys.argv[2], [Path(x) for x in sys.argv[3:]]
tr = load_set(train_file, None)
first = [e for e in tr if not e.meta.get("synthetic")]
random.Random(1).shuffle(first)
val = first[:400]
ms = [DecisionModel.load(d) for d in dirs]
sd = {k: sum(m.state_dict()[k].float() for m in ms) / len(ms) for k in ms[0].state_dict()}
m = ms[0]; m.load_state_dict(sd); m.temperature.fill_(1.0)
tok = AutoTokenizer.from_pretrained(dirs[0] / "backbone")
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); m = m.to(dev)
z = predict_logits(m, tok, _texts(val), max_len=m.max_len, batch_size=32).reshape(-1)
yv = np.array([e.labels[QID] for e in val])
lr = LogisticRegression(C=100).fit(z[:, None], yv)
a, c = float(lr.coef_[0, 0]), float(lr.intercept_[0])
with torch.no_grad():
    m.head.weight.mul_(a); m.head.bias.mul_(a).add_(c)
d = protocol.dev(TASK); g = np.array([e.gold[QID] for e in d])
p = predict_probs(m, tok, d, batch_size=32)
print("soup2", [x.name for x in dirs], f"platt a={a:.2f} c={c:.2f}", "dev_acc", np.mean((p >= .5) == g), "auroc", round(roc_auc_score(g, p), 3), "A", np.mean((p[:100]>=.5)==g[:100]), "B", np.mean((p[100:]>=.5)==g[100:]))
m.cpu(); m.save(out, tok); np.save(out / "dev_probs.npy", p)
