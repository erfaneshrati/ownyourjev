"""Re-predict dev with the exact deployed artifact/inference path of predict_eval.py (final_r2/model.int8.onnx,
OnnxPredictor, threads=4, batch 1) and verify against the frozen test probabilities on a sample of test inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
from ownyourjev.export import OnnxPredictor

md = D / "models/final_r2"
p = OnnxPredictor(md, threads=4)
print("onnx file:", getattr(p, "onnx_path", "model.int8.onnx (default)"))
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
a = np.concatenate([p.probs([e.state]) for e in dev]).astype(np.float64)
print("dev spearman", round(sp(g, a @ np.arange(6)), 4))
old = np.load(R / "dev_probs.npy")
print("max |dev - saved dev_probs.npy| =", float(np.abs(a - old).max()))
ev = protocol.eval_inputs(T); frozen = np.load(R / "eval_probs.npy")
idx = np.random.default_rng(0).choice(len(ev), 50, replace=False)
pe = np.concatenate([p.probs([ev[i].state]) for i in idx]).astype(np.float64)
print("max |test re-pred - frozen| on 50 inputs =", float(np.abs(pe - frozen[idx]).max()))
np.save(R / "dev_probs_deployed.npy", a)
