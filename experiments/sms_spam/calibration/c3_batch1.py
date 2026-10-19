"""Batch-1 serving: recompute test (each text by its own fold model) and dev (all 5 fold models) int8 probs at
batch size 1, with the same artifact / preprocessing / threads (common.onnx_probs, threads=8, batch=1)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
T = "sms_spam"; M = ROOT / "data/v2/sms_spam/models/gte_full_e3"; R = protocol.RESULTS / T
dev = protocol.dev(T); ev = protocol.eval_inputs(T)
out = np.full(len(ev), np.nan, dtype=np.float32); D = np.zeros((5, len(dev)), dtype=np.float32)
for k in range(5):
    d = M / f"fold{k}"; t = time.time()
    te_idx = np.load(d / "te_idx.npy")
    out[te_idx] = onnx_probs(d, [ev[i] for i in te_idx], batch=1, onnx_path=d / "model.int8.onnx")
    D[k] = onnx_probs(d, dev, batch=1, onnx_path=d / "model.int8.onnx")
    print(f"fold{k} done {time.time()-t:.0f}s", flush=True)
assert not np.isnan(out).any()
np.save(ROOT / "data/v2/sms_spam/eval_probs_b1.npy", out)
np.save(R / "dev_probs_b1.npy", D)
p0 = np.load(ROOT / "data/v2/sms_spam/eval_probs.npy"); d0 = np.load(R / "dev_probs_deployed.npy")
print("test max|dp| %.3f mean|dp| %.4f decision flips %d" % (np.abs(out-p0).max(), np.abs(out-p0).mean(), ((out>=.5)!=(p0>=.5)).sum()))
print("dev  max|dp| %.3f mean|dp| %.4f decision flips %d/1000" % (np.abs(D-d0).max(), np.abs(D-d0).mean(), ((D>=.5)!=(d0>=.5)).sum()))
# determinism check: re-run 20 test texts at batch 1
d = M / "fold0"; idx = np.load(d / "te_idx.npy")[:20]
print("batch-1 repeatability max|d|", np.abs(onnx_probs(d, [ev[i] for i in idx], batch=1, onnx_path=d / "model.int8.onnx") - out[idx]).max())
