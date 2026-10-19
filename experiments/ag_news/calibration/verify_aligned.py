"""Re-predict test inputs with the original batch alignment (64-batches from index 0), as finalize_probs.py did."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
T = "ag_news"; D, R = paths(T)
pred = OnnxPredictor(ROOT / "data/v2/ag_news/models/v1plain_5k", threads=8)
ev = protocol.eval_inputs(T); pf = np.load(R / "eval_probs.npy")
starts = [0, 64 * 30, 64 * 60, 64 * 90]
d = []
for s in starts:
    p = pred.probs([e.state for e in ev[s:s + 64]]); d.append(float(np.abs(p - pf[s:s + 64]).max()))
print(json.dumps({"aligned_batch_max_diff": d, "n": 64 * len(starts)}))
