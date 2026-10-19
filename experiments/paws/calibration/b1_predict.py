"""Batch-1 predictions (deployed serving mode) of the frozen partial-int8 model for test (eval_inputs, no labels) and dev.
Same artifact (model.int8.onnx), preprocessing (OnnxPredictor: render_state, max_len 128) and threads (8) as common.onnx_probs."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.export import OnnxPredictor
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
pr = OnnxPredictor(d, onnx_path=d / "model.int8.onnx", threads=8)
def b1(exs): return np.array([pr.probs([e.state]).reshape(-1)[0] for e in exs], dtype=np.float32)
dev = protocol.dev("paws"); pd = b1(dev)
np.save(ROOT / "benchmarks/results_v2/paws/dev_probs_b1.npy", pd)
old = np.load(ROOT / "experiments/paws/calibration/dev_probs_batch1.npy")
print("dev b1 reproducible vs c7 batch1:", float(np.abs(pd - old).max()), flush=True)
ev = protocol.eval_inputs("paws"); pt = b1(ev)
np.save(ROOT / "data/v2/paws/eval_probs_b1.npy", pt)
pt2 = b1(ev[:50]); print("test b1 deterministic (first 50 re-run):", float(np.abs(pt2 - pt[:50]).max()))
fz = np.load(ROOT / "data/v2/paws/eval_probs.npy")
print("vs frozen: mean|diff|", float(np.abs(pt - fz).mean()), "max", float(np.abs(pt - fz).max()), "flip frac", float(((pt >= .5) != (fz >= .5)).mean()))
