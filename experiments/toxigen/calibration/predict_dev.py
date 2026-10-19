"""Re-predict dev with the exact deployed artifact (soup_silv05_s012/model.onnx, fp32, onnx_probs path used by export_eval.py)
and verify on a few eval_inputs against the frozen test probabilities. Writes dev_probs_deployed_{toxic,toxicity}.npy."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from common import DATA, TASK
from mt_model import onnx_probs
from benchmarks.v2 import protocol as P

d = DATA / "models" / "soup_silv05_s012"
R = P.RESULTS / TASK
LV = np.arange(5) / 4.0

def heads(p):
    tox = p[:, 1:] @ LV  # deployed toxic probability = E[toxicity]/4
    return tox, p[:, 1:]

ev = P.eval_inputs(TASK)
idx = list(range(0, len(ev), 47))[:20]
pe = onnx_probs(d, [ev[i].state["text"] for i in idx], "model.onnx")
ft, fs = np.load(R / "eval_probs_toxic.npy"), np.load(R / "eval_probs_toxicity.npy")
t, s = heads(pe)
print("eval check n=%d max|dtoxic|=%.2e max|dtoxicity|=%.2e" % (len(idx), np.abs(t - ft[idx]).max(), np.abs(s - fs[idx]).max()))

dev = P.dev(TASK)
pd = onnx_probs(d, [e.state["text"] for e in dev], "model.onnx")
old = np.load(d / "dev_probs_fp32onnx.npy")
print("vs saved dev_probs_fp32onnx max diff %.2e" % np.abs(pd - old).max())
t, s = heads(pd)
np.save(R / "dev_probs_deployed_toxic.npy", t); np.save(R / "dev_probs_deployed_toxicity.npy", s)
print("saved", t.shape, s.shape)
