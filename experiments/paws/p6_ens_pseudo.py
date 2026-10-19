"""Average two teachers' pseudo-labels (seeds 0,1) -> pseudo_" + sys.argv[1] + ".jsonl"""
import sys
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs, pick_device
from transformers import AutoTokenizer
T = "paws"
have = set(load_v1(T, "silver_seeded")) | {e.uid for e in load_jsonl(ROOT / "data/v2/paws/v2_seeded_more.jsonl")}
rest = [e for e in protocol.pool(T) if e.uid not in have]
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
P, PD = [], []
for n in sys.argv[2].split(","):
    d = ROOT / "data/v2/paws/models" / n
    m = DecisionModel.load(d, device=pick_device(6.0)); tok = AutoTokenizer.from_pretrained(d / "backbone")
    P.append(predict_probs(m, tok, rest, batch_size=128).reshape(-1)); PD.append(np.load(d / "dev_probs.npy"))
    del m
p = np.mean(P, 0); pd_ = np.mean(PD, 0)
print("teacher-ensemble dev acc", acc_f1(y, pd_)[0], "teacher agreement on pool", float(((P[0] >= .5) == (P[1] >= .5)).mean()))
save_jsonl(ROOT / f"data/v2/paws/pseudo_{sys.argv[1]}.jsonl", [Example(e.uid, e.state, {"answer": bool(pi >= 0.5)}, None, "", {"soft": {"answer": float(pi)}, "src": "pseudo:" + sys.argv[1]}) for e, pi in zip(rest, p)])
