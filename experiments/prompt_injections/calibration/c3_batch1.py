"""Batch-1 serving: deployed int8 on test, 5 CV-fold int8 models on their held-out dev folds, one text per call; refit calibrator."""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.data import kfold
from ownyourjev.export import OnnxPredictor
from ownyourjev.calibration import select_calibrator
T = "prompt_injections"; D = ROOT / "data/v2" / T; R = protocol.RESULTS / T

def b1(model_dir, exs):
    pr = OnnxPredictor(model_dir, threads=8)  # same predictor + threads as onnx_probs
    return np.array([float(pr.probs([e.state]).reshape(-1)[0]) for e in exs], dtype=np.float32)

ev = protocol.eval_inputs(T)
pe = b1(D / "models/final_gte_e8_dup3", ev); np.save(D / "eval_probs_b1.npy", pe)
print("test max|d| vs frozen", float(np.abs(pe - np.load(D / "eval_probs.npy")).max()), "flips", int(((pe >= .5) != (np.load(D / "eval_probs.npy") >= .5)).sum()), flush=True)
dev = protocol.dev(T); y = np.array([int(bool(e.labels["answer"])) for e in dev]); pd = np.zeros(len(dev))
for k, (tr_i, te_i) in enumerate(kfold(dev, 5, seed=0)):
    pd[te_i] = b1(D / f"models/cv_gte_e8_dup3/fold{k}", [dev[i] for i in te_i])
np.save(D / "dev_probs_b1.npy", pd); np.save(R / "dev_probs_b1.npy", pd)
old = np.load(D / "oof_cv_gte_e8_dup3_int8.npy")
print("dev max|d| vs batched oof", float(np.abs(pd - old).max()), "dev acc b1", acc_f1(y, pd)[0], "batched", acc_f1(y, old)[0])
cal, scores = select_calibrator(pd, y)
for m, s in scores.items(): print(f"{m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f} acc {s['accuracy']:.3f}")
print("chosen", cal.to_dict())
out = {"description": "Batch-1 recompute (one text per OnnxPredictor.probs call, same int8 artifacts, preprocessing, threads=8). Test: deployed final_gte_e8_dup3/model.int8.onnx on protocol.eval_inputs. Dev: out-of-fold, each of the 5 CV fold int8 models (identical recipe, dev fold held out) on its held-out fold. Calibrator refit by select_calibrator (5-fold CV NLL) on the batch-1 dev OOF probs.",
       "heads": {"answer": {"probs": f"data/v2/{T}/eval_probs_b1.npy", "dev_probs": f"data/v2/{T}/dev_probs_b1.npy"}},
       "calibration": {"answer": cal.to_dict()},
       "dev_cv": {"answer": scores}}
(R / "BATCH1.json").write_text(json.dumps(out, indent=1, default=float))
