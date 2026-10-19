"""(First run raised at log_experiment on an invalid outcome string after CALIBRATION.json was written; ledger line logged by c9.)
Fit on dev only (deployed-path dev probs), select by 5-fold CV NLL, freeze CALIBRATION.json, log ledger line."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.calibration import select_calibrator
from benchmarks.v2.ledger import log_experiment
R = ROOT / "benchmarks/results_v2/paws"
dev = protocol.dev("paws"); y = np.array([int(e.labels["answer"]) for e in dev])
p = np.load(R / "dev_probs_deployed.npy")
cal, sc = select_calibrator(p, y)
desc = ("Dev predictions: deployed partial-int8 ONNX (model.int8.onnx, identical md5 to model.int8_noattn_nodown.onnx) via "
        "experiments/paws/common.onnx_probs (length-sorted batches of 32, 8 threads); bit-identical to the p4 dev probs behind "
        "FROZEN dev_metrics.accuracy_int8=0.885. Dev is out-of-sample: p2_train.labelset drops all seed and dev uids and the "
        "pseudo-labels cover pool texts only. Re-predicting test did NOT reproduce the frozen test probs to 1e-4 (mean |diff| 0.013, "
        "corr 0.997, 98.7% decision agreement): dynamic int8 activation scales depend on batch composition and the exact test "
        "batching is unrecorded. The same noise level on dev (batch-1 vs batch-32: mean |diff| 0.012) leaves the selection unchanged "
        "(platt both times, a 0.527 vs 0.526, b 0.831 vs 0.824). Method chosen by 5-fold CV NLL on the 200 dev examples, refit on all 200.")
out = {"description": desc, "heads": {"answer": {**cal.to_dict(), "dev_cv": sc, "n_fit": int(len(y))}}}
(R / "CALIBRATION.json").write_text(json.dumps(out, indent=1, default=float))
print(cal.to_dict())
summ = {m: {"nll": round(s["nll"], 4), "ece": round(s["ece"], 4), "brier": round(s["brier"], 4)} for m, s in sc.items()}
log_experiment(task="paws", round=99, kind="selection", idea="post-hoc calibration", setting="calibration",
               dev={"chosen": cal.method, "params": cal.params, "dev_cv": summ, "n_fit": 200},
               outcome="success", takeaway=f"chosen {cal.method} by 5-fold CV NLL on dev ({summ[cal.method]['nll']} vs none {summ['none']['nll']})")
