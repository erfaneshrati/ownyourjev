"""Sensitivity of calibration to int8 batch-composition noise, then dev-only method selection (5-fold CV, lowest NLL)."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator, Calibrator
d = ROOT / "data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0"
dev = protocol.dev("paws"); y = np.array([int(e.labels["answer"]) for e in dev])
p = np.load(ROOT / "benchmarks/results_v2/paws/dev_probs_deployed.npy")
pr = OnnxPredictor(d, onnx_path=d / "model.int8.onnx", threads=8)
b1 = np.array([pr.probs([e.state]).reshape(-1)[0] for e in dev])
np.save(ROOT / "experiments/paws/calibration/dev_probs_batch1.npy", b1)
print("dev batch1 vs deployed batching: mean|diff|", float(np.abs(b1 - p).mean()), "max", float(np.abs(b1 - p).max()), "acc", acc_f1(y, b1)[0])
# label-free test confidence distributions: frozen vs full re-prediction
fz = np.load(ROOT / "data/v2/paws/eval_probs.npy"); rp = np.load(ROOT / "experiments/paws/calibration/test_repred.npy")
for nm, q in [("frozen", fz), ("repred", rp)]:
    c = np.maximum(q, 1 - q); print(nm, "mean conf", c.mean(), "frac conf>0.99", (c > .99).mean(), "pos rate", (q >= .5).mean())
for nm, q in [("deployed", p), ("batch1", b1)]:
    cal, sc = select_calibrator(q, y)
    print(nm, "chosen", cal.method, cal.params if cal.method != "isotonic" else "(isotonic)")
    for m, s in sc.items(): print(f"   {m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f}")
    for m in ("temperature", "platt"): print("   full-dev fit", m, Calibrator(m).fit(q, y).params)
