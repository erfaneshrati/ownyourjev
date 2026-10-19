"""Out-of-sample dev predictions from the exact deployed artifact and inference path that produced the frozen
test probabilities (predict_eval.py: soupAB_sq0.75/model.int8.onnx, batch 1, max_len 512, 8 threads).
Verifies the path by re-predicting the first 60 eval inputs (no labels) against the frozen test probs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import *
from mh import onnx_probs, texts_of
name = "soupAB_sq0.75"
md = D / "models" / name
frozen = json.loads((R / "FROZEN.json").read_text())
ev = protocol.eval_inputs(T)[:60]
pe = onnx_probs(md, texts_of(ev), "model.int8.onnx", threads=8)
for i, h in enumerate(HEADS):
    ref = np.load(ROOT / frozen["heads"][h]["probs"])[:60]
    print(f"verify {h}: max|dp| vs frozen test probs = {np.abs(pe[:, i] - ref).max():.2e}")
dev = protocol.dev(T)
pd = onnx_probs(md, texts_of(dev), "model.int8.onnx", threads=8)
old = np.load(D / "runs" / name / "dev_probs_int8.npy")
print(f"vs existing dev_probs_int8.npy (threads=4): max|dp| = {np.abs(pd - old).max():.2e}")
for i, h in enumerate(HEADS):
    np.save(R / f"dev_probs_deployed_{h}.npy", pd[:, i].astype(np.float64))
    np.save(D / "runs" / name / f"dev_probs_deployed_{h}.npy", pd[:, i].astype(np.float64))
    y = np.array([int(bool(e.gold[h])) for e in dev])
    print(h, "n_pos", y.sum(), "mean p", pd[:, i].mean().round(4))
