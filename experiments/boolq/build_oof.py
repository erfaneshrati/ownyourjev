"""Round 3: out-of-fold teacher probabilities for every pool text (5-fold cross-fit soft-label teachers).
soft target = w*OOF_teacher + (1-w)*vote soft label."""
import sys, json, numpy as np, torch
from common import *
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from transformers import AutoTokenizer
w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
out, agree = [], []
cache = DATA / "cv" / "oof_probs.json"
oof = json.loads(cache.read_text()) if cache.is_file() else {}
import json as _j
for k in range(5):
    ho = read_jsonl(DATA / f"cv/fold{k}_heldout.jsonl")
    if not all(e.uid in oof for e in ho):
        d = DATA / f"models/cvT{k}"
        m = DecisionModel.load(d, device=dev)
        p = predict_probs(m, AutoTokenizer.from_pretrained(d / "backbone"), ho, batch_size=64)
        oof.update({e.uid: float(x) for e, x in zip(ho, p)})
        del m; torch.cuda.empty_cache()
    for e in ho:
        t = oof[e.uid]; v = e.meta["soft"][QID]
        s = w * t + (1 - w) * v
        agree.append((t >= .5) == (v >= .5))
        out.append(Example(e.uid, e.state, {QID: s >= .5}, None, "", {"soft": {QID: s}, "oof": t, "vote": v}))
cache.write_text(_j.dumps(oof))
write_jsonl(DATA / f"pool_oof_w{w}.jsonl", out)
print("n", len(out), "OOF teacher vs vote agreement", float(np.mean(agree)))
