"""Batch-size-1 predictions of the frozen int8 model (one text per OnnxPredictor.probs call, threads=8 as in
finalize_probs.py), dev + eval_inputs; refit calibrator (none/temperature) on batch-1 dev; write BATCH1.json."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator, cv_scores
T = "ag_news"; D, R = paths(T)
frozen = json.loads((R / "FROZEN.json").read_text())
pred = OnnxPredictor(ROOT / frozen["model_dir"], threads=8)
q = protocol.task_spec(T).parsed["answer"]
def run(exs, tag):
    out, t0 = [], time.time()
    for i, e in enumerate(exs):
        out.append(pred.probs([e.state])[0])
        if i % 1000 == 0:
            print(tag, i, f"{time.time() - t0:.0f}s", flush=True)
    return np.stack(out)
dev = protocol.dev(T)
pd = run(dev, "dev"); np.save(R / "dev_probs_b1.npy", pd)
pe = run(protocol.eval_inputs(T), "eval"); np.save(R / "eval_probs_b1.npy", pe)
# determinism check: repeat a few at batch 1
rep = np.stack([pred.probs([e.state])[0] for e in dev[:20]])
y = np.array([q.options.index(e.labels["answer"]) for e in dev])
cal, cv = select_calibrator(pd, y, methods=("none", "temperature"))
full = cv_scores(pd, y)
pd0 = np.load(R / "dev_probs_deployed_answer.npy"); pe0 = np.load(R / "eval_probs.npy")
info = {"repeat_max_diff": float(np.abs(rep - pd[:20]).max()), "dev_acc_b1": float((pd.argmax(1) == y).mean()),
        "dev_acc_batched": float((pd0.argmax(1) == y).mean()), "dev_flips": int((pd.argmax(1) != pd0.argmax(1)).sum()),
        "dev_max_abs_diff": float(np.abs(pd - pd0).max()), "eval_flips_vs_frozen": int((pe.argmax(1) != pe0.argmax(1)).sum()),
        "eval_max_abs_diff": float(np.abs(pe - pe0).max())}
out = {"description": ("Frozen int8 model data/v2/ag_news/models/v1plain_5k/model.int8.onnx via OnnxPredictor(threads=8), one text per "
       "call (no padding), on protocol.dev and protocol.eval_inputs. Calibrator refit on batch-1 dev predictions (model never "
       "trained on dev), 5-fold CV NLL over argmax-preserving none/temperature, refit on all 200 dev."),
       "heads": {"answer": {"probs": "benchmarks/results_v2/ag_news/eval_probs_b1.npy",
                            "dev_probs": "benchmarks/results_v2/ag_news/dev_probs_b1.npy"}},
       "calibration": {"answer": {**cal.to_dict(), "dev_cv": cv, "dev_cv_unrestricted": full, "n_fit": int(len(y))}},
       "checks": info}
(R / "BATCH1.json").write_text(json.dumps(out, indent=1, default=float))
print("INFO", json.dumps(info)); print("CAL", cal.to_dict())
for m, s in full.items():
    print(f"{m:12s} nll={s['nll']:.4f} ece={s['ece']:.4f} ace={s['ace']:.4f} brier={s['brier']:.4f} conf={s['mean_conf']:.3f}")
print("DONE")
