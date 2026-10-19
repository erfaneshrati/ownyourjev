"""Probe which batching reproduces the frozen test probs (on the first n test inputs, no labels)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.export import OnnxPredictor
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
ev = protocol.eval_inputs("paws"); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
n = 64
pr = OnnxPredictor(d, onnx_path=d / "model.int8.onnx", threads=8)
b1 = np.array([pr.probs([e.state]).reshape(-1)[0] for e in ev[:n]])
print("batch1 max|diff|", float(np.abs(b1 - frozen[:n]).max()))
for bs in (16, 32, 64):
    seq = np.concatenate([pr.probs([e.state for e in ev[s:s + bs]]).reshape(-1) for s in range(0, n, bs)])
    print(f"unsorted bs{bs} max|diff|", float(np.abs(seq - frozen[:n]).max()))
