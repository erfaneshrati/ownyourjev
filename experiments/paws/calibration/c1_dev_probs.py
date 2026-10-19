"""Out-of-sample dev predictions of the deployed PAWS model (dev never in training: p2_train.labelset drops seeds+dev uids).
Same artifact (model.int8.onnx == model.int8_noattn_nodown.onnx, partial int8) and same inference path (common.onnx_probs)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
T = "paws"
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
onnx = d / "model.int8.onnx"
dev = protocol.dev(T); y = np.array([int(e.labels["answer"]) for e in dev])
p = onnx_probs(d, dev, onnx_path=onnx)
old = np.load(d / "dev_probs_int8_noattn_nodown.npy")
print("dev acc", acc_f1(y, p)[0], "max |diff| vs saved p4 dev probs", float(np.abs(p - old).max()))
ev = protocol.eval_inputs(T); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
pt = onnx_probs(d, ev[:n], onnx_path=onnx)
print(f"test first {n}: max |diff| vs frozen", float(np.abs(pt - frozen[:n]).max()), "mean", float(np.abs(pt - frozen[:n]).mean()))
np.save(ROOT / "benchmarks/results_v2/paws/dev_probs_deployed.npy", p)
