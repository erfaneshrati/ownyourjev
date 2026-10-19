"""Export each fold model to ONNX int8, score dev (int8 and fp32) and predict the fold's eval texts with the int8 model."""
import json, sys
import numpy as np
from common import *
import export_mh as X
ev = P.eval_inputs(T); folds = P.eval_folds(T); dev = P.dev(T)
PFX = sys.argv[1] if len(sys.argv) > 1 else "final_f"; EP = sys.argv[2] if len(sys.argv) > 2 else "F"
res = {}
for k in range(5):
    d = DATA / "models" / f"{PFX}{k}{__import__('os').environ.get('SFX', '')}"
    if not (d / "model.onnx").exists():
        X.export(d)
    else:
        X.quantize(d)
    out = {}
    for q in (True, False):
        pr = X.Predictor(d, threads=8, int8=q)
        p = np.concatenate([pr.probs([e.state]) for e in dev])
        r = score_dev({dd: p[:, j] @ np.arange(5) for j, dd in enumerate(DIMS)}, dev)
        out["int8" if q else "fp32"] = {kk: round(v, 4) for kk, v in r.items()}
        if q:
            pe = np.concatenate([pr.probs([ev[i].state]) for i in folds[k][1]])
            np.save(DATA / "eval_preds" / f"{EP}{k}_int8.npy", pe); np.save(DATA / "dev_hold" / f"{EP}{k}_int8_dev.npy", p)
    res[f"fold{k}"] = out; print(k, out, flush=True)
(RES / f"dev_onnx_folds_{EP}.json").write_text(json.dumps(res, indent=1))
