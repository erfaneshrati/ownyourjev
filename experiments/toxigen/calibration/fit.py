"""Select and fit per-head calibrators on the 200 dev examples (5-fold CV over none/temperature/platt|vector/isotonic,
lowest NLL), then freeze benchmarks/results_v2/toxigen/CALIBRATION.json. Reads no test labels."""
import json
import numpy as np
from benchmarks.tasks import get_task
from benchmarks.v2 import protocol as P
from benchmarks.v2.calibration_eval import gold_index
from ownyourjev.calibration import select_calibrator

TASK = "toxigen"
R = P.RESULTS / TASK
spec = get_task(TASK).spec
dev = P.dev(TASK)
out = {"description": (
    "Dev predictions from the exact deployed artifact (data/v2/toxigen/models/soup_silv05_s012/model.onnx, fp32, "
    "experiments/toxigen/mt_model.py:onnx_probs, max_len 64); dev was never trained on (pool excludes dev+seed uids, "
    "silver filtered by the same exclusion), dev only chose config and the 0.30 threshold. Verified: re-predicting 20 "
    "eval_inputs reproduces the frozen test probs exactly (max diff 0). toxic = E[toxicity]/4 (a score, not a "
    "probability); toxicity = 5-level softmax. Each head calibrated separately with select_calibrator (5-fold CV on "
    "200 dev, lowest NLL, refit on all dev). Decisions unchanged (threshold 0.30 applies to raw probs)."),
    "heads": {}}
for h in ["toxic", "toxicity"]:
    q = spec.parsed[h]
    p = np.load(R / f"dev_probs_deployed_{h}.npy")
    y = np.array([gold_index(q, e.gold[h]) for e in dev])
    cal, cv = select_calibrator(p, y)
    print(h, "counts", np.bincount(y), "->", cal.method, cal.params)
    for m, r in cv.items():
        print(f"  {m:12s} nll {r['nll']:.4f} ece {r['ece']:.4f} ace {r['ace']:.4f} brier {r['brier']:.4f}")
    out["heads"][h] = {**cal.to_dict(), "dev_cv": cv, "n_fit": len(y)}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
