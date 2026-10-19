"""Batch-1: refit calibrator on batch-1 dev predictions (pooled 5 fold models x 200, grouped CV) and write BATCH1.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa
from ownyourjev.calibration import select_calibrator
T = "sms_spam"; R = protocol.RESULTS / T
dev = protocol.dev(T); y = np.array([int(bool(e.labels["answer"])) for e in dev])
D = np.load(R / "dev_probs_b1.npy").astype(float)
cal, cv = select_calibrator(D.reshape(-1), np.tile(y, 5), methods=("none", "temperature", "platt", "isotonic"), groups=np.tile(np.arange(len(y)), 5))
for m, v in cv.items(): print(f"{m:12s} nll {v['nll']:.4f} ece {v['ece']:.4f} ace {v['ace']:.4f} brier {v['brier']:.4f}")
print("->", cal.to_dict())
from common import acc_f1
print("per-fold dev F1 @0.5 batch1:", [round(acc_f1(y, D[k])[1], 4) for k in range(5)])
spec = {"description": "Batch-size-1 int8 ONNX predictions (deployed serving mode): same 5 fold artifacts, preprocessing, "
        "max_len 128, 8 threads, each test text scored by its own fold model (te_idx), one text per call. Dev: all 5 fold models "
        "at batch 1 (dev is in no fold's training). Calibrator refit on batch-1 dev, pooled 1000 rows, 5-fold CV grouped by dev text, lowest NLL.",
        "heads": {"answer": {"probs": "data/v2/sms_spam/eval_probs_b1.npy", "dev_probs": "benchmarks/results_v2/sms_spam/dev_probs_b1.npy"}},
        "calibration": {"answer": {**cal.to_dict(), "dev_cv": cv, "n_fit": int(D.size)}}}
(R / "BATCH1.json").write_text(json.dumps(spec, indent=1, default=float))
