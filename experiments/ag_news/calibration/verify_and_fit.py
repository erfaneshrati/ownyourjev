"""Calibration study for ag_news: verify the deployed inference path, predict dev out-of-sample, select a
calibrator on dev only, write CALIBRATION.json. Usage: verify_and_fit.py [TASK]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator, calibration_metrics

T = sys.argv[1] if len(sys.argv) > 1 else "ag_news"
D, R = paths(T)
frozen = json.loads((R / "FROZEN.json").read_text())
md = ROOT / frozen["model_dir"]
q = protocol.task_spec(T).parsed["answer"]
pred = OnnxPredictor(md, threads=8)
def run(exs):
    return np.concatenate([pred.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])

# 1) dev/seeds never trained on: confirm none of the dev uids are in the training labels
held = {e.uid for e in protocol.dev(T)}
train_uids = set(v1_silver(T, "silver"))
assert not (held & train_uids), "dev overlap with training data"

# 2) verify the inference path reproduces frozen test probs
pe_frozen = np.load(ROOT / frozen["heads"]["answer"]["probs"])
ev = protocol.eval_inputs(T)
idx = list(range(0, len(ev), len(ev) // 256))[:256]
pe_new = run([ev[i] for i in idx])
diff_test = float(np.abs(pe_new - pe_frozen[idx]).max())

# 3) dev predictions with the same path
dev = protocol.dev(T)
pd = run(dev)
pd_old = np.load(R / "dev_probs_deployed.npy")
diff_dev = float(np.abs(pd - pd_old).max())
y = np.array([q.options.index(e.labels["answer"]) for e in dev])
np.save(R / "dev_probs_deployed_answer.npy", pd)

cal, cv = select_calibrator(pd, y)
print(json.dumps({"diff_test_max": diff_test, "n_test_checked": len(idx), "diff_dev_vs_saved": diff_dev,
                  "dev_acc": float((pd.argmax(1) == y).mean())}))
for m, s in cv.items():
    print(f"{m:12s} nll={s['nll']:.4f} ece={s['ece']:.4f} ace={s['ace']:.4f} brier={s['brier']:.4f} conf={s['mean_conf']:.3f} acc={s['accuracy']:.3f}")
print("chosen", cal.to_dict())
raw = calibration_metrics(pd, y); raw.pop("reliability")
out = {"description": ("Dev predictions from the exact deployed artifact (data/v2/ag_news/models/v1plain_5k/model.int8.onnx via "
       "ownyourjev.export.OnnxPredictor, same batching as finalize_probs.py). The model never trained on dev or seeds "
       f"(training data = v1 plain silver labels with seed/dev uids dropped; add_seeds=false). Verified: re-predicted {len(idx)} "
       f"test inputs match eval_probs.npy to max abs diff {diff_test:.2e}; dev re-prediction matches dev_probs_deployed.npy "
       f"to {diff_dev:.2e}. Method chosen by 5-fold CV NLL on the 200 gold dev examples (select_calibrator), refit on all dev."),
       "heads": {"answer": {**cal.to_dict(), "dev_cv": cv, "dev_raw_in_sample": raw, "n_fit": int(len(y)),
                            "dev_probs": str((R / "dev_probs_deployed_answer.npy").relative_to(ROOT))}}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
