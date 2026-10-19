"""Refit the linear head of a soup on silver (teacher soft labels -> LR on argmax with soft-prob sample weights)."""
import sys; sys.path.insert(0,"experiments/sst5")
import numpy as np, torch
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import embed_texts, _texts, predict_probs
from lab import *
name = sys.argv[1]; d = f"data/v2/sst5/models/{name}"
m = DecisionModel.load(d, device="cuda:1"); tok = AutoTokenizer.from_pretrained(d + "/backbone")
L = load("pool_v1_deepseek-v4.1-flash_9000"); excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
pool = [e for e in P.pool(T) if e.uid not in excl and L.get(e.uid, {}).get("soft")]
X = embed_texts(m, tok, _texts(pool), max_len=128, use_cache=False); S = np.array([L[e.uid]["soft"] for e in pool])
Xd = embed_texts(m, tok, _texts(DEV), max_len=128, use_cache=False)
# soft labels: replicate each row per class with weight = prob
Xr = np.repeat(X, 5, 0); yr = np.tile(np.arange(5), len(X)); wr = S.reshape(-1)
keep = wr > 0.01
for C in [0.01, 0.1, 1.0]:
    clf = LogisticRegression(C=C, max_iter=2000).fit(Xr[keep], yr[keep], sample_weight=wr[keep])
    pd = clf.predict_proba(Xd); print("C", C, end=" "); report(f"{name}-refit", pd)
