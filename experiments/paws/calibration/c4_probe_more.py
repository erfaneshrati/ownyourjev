"""Probe more candidate paths against the frozen test probs (first n test inputs, no labels)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.export import OnnxPredictor
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
ev = protocol.eval_inputs("paws"); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
n = 64
fp = OnnxPredictor(d, onnx_path=d / "model.onnx", threads=8)
f1 = np.array([fp.probs([e.state]).reshape(-1)[0] for e in ev[:n]])
print("fp32 onnx batch1 max|diff|", float(np.abs(f1 - frozen[:n]).max()), "mean", float(np.abs(f1 - frozen[:n]).mean()))
pr = OnnxPredictor(d, onnx_path=d / "model.int8.onnx", threads=8)
for bs in (128, 256):
    seq = pr.probs([e.state for e in ev[:bs]]).reshape(-1)
    print(f"int8 unsorted bs{bs} max|diff| on first {bs}", float(np.abs(seq - frozen[:bs]).max()), "mean", float(np.abs(seq - frozen[:bs]).mean()))
