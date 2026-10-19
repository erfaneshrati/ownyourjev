"""Assemble cross-fitted eval probabilities (fold k model -> fold k texts) and write FROZEN.json."""
import json, sys
import numpy as np
from common import *
EP = __import__("os").environ.get("EP", "F")
ev = P.eval_inputs(T); folds = P.eval_folds(T)
probs = np.full((len(ev), 4, 5), np.nan, dtype=np.float32)
for k, (_, te) in enumerate(folds):
    p = np.load(DATA / "eval_preds" / f"{EP}{k}_int8.npy"); assert p.shape == (len(te), 4, 5)
    probs[te] = p
assert not np.isnan(probs).any()
out = DATA / ("final_eval_probs" if EP == "F" else f"final_eval_probs_{EP}"); out.mkdir(exist_ok=True)
heads = {}
for j, d in enumerate(DIMS):
    np.save(out / f"{d}.npy", probs[:, j]); heads[d] = {"probs": str(out / f"{d}.npy"), "threshold": 0.5}
dev_metrics = json.loads((RES / "dev_metrics.json").read_text())
frozen = {
    "description": sys.argv[1],
    "heads": heads,
    "dev_metrics": dev_metrics,
    "latency": str(RES / "latency.json"),
    "label_spend_usd": float(sys.argv[2]),
}
(RES / "FROZEN.json").write_text(json.dumps(frozen, indent=1)); print(json.dumps(frozen, indent=1)[:3000])
