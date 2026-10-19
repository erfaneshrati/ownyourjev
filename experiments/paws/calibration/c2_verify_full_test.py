"""Verify the inference path: re-predict ALL eval_inputs (no labels) with common.onnx_probs on the deployed model.int8.onnx
(length-sorted batches of 32 over the whole set, as for the frozen file). Dynamic int8 activation scales are per batch, so
subset re-prediction changes batch composition; the full set reproduces it."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
ev = protocol.eval_inputs("paws"); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
pt = onnx_probs(d, ev, onnx_path=d / "model.int8.onnx")
diff = np.abs(pt - frozen)
print("n", len(ev), "max |diff|", float(diff.max()), "mean", float(diff.mean()), "frac>1e-4", float((diff > 1e-4).mean()),
      "decision agreement", float(((pt >= .5) == (frozen >= .5)).mean()))
np.save(ROOT / "experiments/paws/calibration/test_repred.npy", pt)
