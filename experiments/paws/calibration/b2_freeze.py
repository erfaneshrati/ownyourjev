"""Refit the calibrator on batch-1 DEV predictions (5-fold CV NLL over none/temperature/platt/isotonic), write BATCH1.json."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.calibration import select_calibrator
from benchmarks.v2.ledger import log_experiment
R = ROOT / "benchmarks/results_v2/paws"
dev = protocol.dev("paws"); y = np.array([int(e.labels["answer"]) for e in dev])
p = np.load(R / "dev_probs_b1.npy")
print("dev b1 acc", acc_f1(y, p)[0])
cal, sc = select_calibrator(p, y, methods=("none", "temperature", "platt", "isotonic"))
for m, s in sc.items(): print(f"  {m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f}")
print("chosen", cal.to_dict())
out = {"description": ("Batch-1 (serving mode) predictions of the frozen partial-int8 UltraBERT (model.int8.onnx, OnnxPredictor, "
                       "8 threads, one text per call) for eval_inputs and dev (experiments/paws/calibration/b1_predict.py). "
                       "Dev is out-of-sample (never trained on). Calibrator refit on the batch-1 dev probs by 5-fold CV NLL."),
       "heads": {"answer": {"probs": "data/v2/paws/eval_probs_b1.npy", "dev_probs": "benchmarks/results_v2/paws/dev_probs_b1.npy",
                            "dev_accuracy_b1": acc_f1(y, p)[0]}},
       "calibration": {"answer": {**cal.to_dict(), "dev_cv": sc, "n_fit": int(len(y))}}}
(R / "BATCH1.json").write_text(json.dumps(out, indent=1, default=float))
summ = {m: {k: round(s[k], 4) for k in ("nll", "ece", "brier")} for m, s in sc.items()}
log_experiment(task="paws", round=99, kind="selection", idea="batch-1 serving recalibration", setting="calibration_batch1",
               dev={"dev_accuracy_b1": acc_f1(y, p)[0], "chosen": cal.method, "params": cal.params, "dev_cv": summ}, outcome="success",
               takeaway=f"{cal.method} refit on batch-1 dev probs")
