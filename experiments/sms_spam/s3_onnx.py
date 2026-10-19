"""Export each fold to ONNX int8; compare dev probs torch vs int8; write int8 eval probs."""
import sys
from common import *
T = "sms_spam"
name = sys.argv[1]; folds = [int(x) for x in sys.argv[2].split(",")]
dev = protocol.dev(T); y = [e.labels["answer"] for e in dev]; ev = protocol.eval_inputs(T)
for k in folds:
    d = ROOT / "data/v2/sms_spam/models" / name / f"fold{k}"
    if not (d / "model.int8.onnx").is_file():
        export_int8(d)
    pq = onnx_probs(d, dev); pt = np.load(d / "dev_probs.npy")
    np.save(d / "dev_probs_int8.npy", pq)
    te_idx = np.load(d / "te_idx.npy")
    np.save(d / "te_probs_int8.npy", onnx_probs(d, [ev[i] for i in te_idx]))
    print(f"fold{k}: torch {acc_f1(y, pt)}  int8 {acc_f1(y, pq)}  max|diff| {np.abs(pq-pt).max():.3f}", flush=True)
