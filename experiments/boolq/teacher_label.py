"""Round 2 (d): ensemble-teacher soft labels on synthetic questions the teachers never trained on.
usage: teacher_label.py OUT_JSONL IN_JSONL[,IN_JSONL...] MODEL_DIR...
soft target = mean(teacher ensemble prob, LLM label) ; keeps only items with an LLM label."""
import sys, numpy as np, torch
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from transformers import AutoTokenizer
out, ins, dirs = Path(sys.argv[1]), sys.argv[2].split(","), [Path(x) for x in sys.argv[3:]]
gu = gold_uids()
exs = [e for p in ins for e in read_jsonl(p) if e.labels is not None and e.meta.get("src_uid") not in gu]
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
P = []
for d in dirs:
    m = DecisionModel.load(d, device=dev)
    P.append(predict_probs(m, AutoTokenizer.from_pretrained(d / "backbone"), exs, batch_size=64))
    del m; torch.cuda.empty_cache()
pt = np.mean(P, 0)
llm = np.array([float(e.labels[QID]) for e in exs])
soft = 0.5 * pt + 0.5 * llm
for e, s, t in zip(exs, soft, pt):
    e.meta["soft"] = {QID: float(s)}; e.meta["teacher"] = float(t); e.labels = {QID: bool(s >= 0.5)}
write_jsonl(out, exs)
print("n", len(exs), "teacher-LLM agreement", float(np.mean((pt >= .5) == (llm >= .5))), "mean teacher conf", float(np.mean(np.maximum(pt, 1 - pt))))
