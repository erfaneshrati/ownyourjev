"""Does a labelled sibling response (same prompt) in the training set inflate dev Spearman? Split dev by whether the
other response to its prompt is among the flash-labelled training examples."""
import numpy as np
from common import *
from postproc import blend_hc
M = ROOT / "data/v2/helpsteer2/models"
dev = protocol.dev(T); G = np.array([[e.gold[a] for a in A] for e in dev], float)
tr = load_jsonl(LD / "silver_v1.jsonl") + load_jsonl(LD / "flash_more.jsonl")
tp = {e.state["prompt"] for e in tr}
has = np.array([e.state["prompt"] in tp for e in dev])
print("dev with labelled sibling in training:", has.sum(), "without:", (~has).sum())
for m in ["F7_s0", "F7_s1", "D_s0"]:
    P = blend_hc(np.load(M / m / "dev_probs10.npy")[:, :5]); E = P @ np.arange(5)
    for nm, ix in [("sibling", has), ("no-sibling", ~has)]:
        rs = [sp(E[ix, j], G[ix, j]) for j in range(5)]
        print(m, f"{nm:10s}", " ".join(f"{r:.3f}" for r in rs), f"mean {np.mean(rs):.3f}")
