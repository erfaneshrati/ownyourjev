"""Calibration step 1: out-of-sample dev predictions from the deployed int8 ONNX fold models + verification.

Dev (200 gold) was carved out of the eval set and is not in any fold's training data, so every fold model's dev
prediction is out-of-sample. Uses the same inference function (common.onnx_probs, length-sorted batches of 32,
8 threads) that produced the frozen test probabilities (s3_onnx.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
T = "sms_spam"; M = ROOT / "data/v2/sms_spam/models/gte_full_e3"
dev = protocol.dev(T); ev = protocol.eval_inputs(T); folds = protocol.eval_folds(T)
frozen = np.load(ROOT / "data/v2/sms_spam/eval_probs.npy")
P = np.zeros((5, len(dev)), dtype=np.float32)
for k in range(5):
    d = M / f"fold{k}"
    P[k] = onnx_probs(d, dev, onnx_path=d / "model.int8.onnx")
    old = np.load(d / "dev_probs_int8.npy")
    te_idx = np.load(d / "te_idx.npy")
    # re-predict the full held-out fold (same batching as s3_onnx) and a 64-example subset
    full = onnx_probs(d, [ev[i] for i in te_idx], onnx_path=d / "model.int8.onnx")
    sub = te_idx[:64]
    ps = onnx_probs(d, [ev[i] for i in sub], onnx_path=d / "model.int8.onnx")
    print(f"fold{k}: dev vs stored dev_probs_int8 max|d| {np.abs(P[k]-old).max():.2e} | test fold full re-predict vs frozen max|d| "
          f"{np.abs(full-frozen[te_idx]).max():.2e} | 64-subset max|d| {np.abs(ps-frozen[sub]).max():.2e}", flush=True)
np.save(ROOT / "data/v2/sms_spam/calibration_dev_probs_all_folds.npy", P)
