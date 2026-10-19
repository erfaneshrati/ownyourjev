"""Verify (1) deployed int8 path reproduces frozen test probs; (2) saved int8 OOF dev probs reproduce from fold int8 models."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from ownyourjev.data import kfold
T = "prompt_injections"; D = ROOT / "data/v2" / T
ev = protocol.eval_inputs(T); frozen = np.load(D / "eval_probs.npy")
idx = list(range(0, len(ev), 7))
p = onnx_probs(D / "models/final_gte_e8_dup3", [ev[i] for i in idx])
print("test re-predict max|diff|", float(np.abs(p - frozen[idx]).max()), "n", len(idx))
dev = protocol.dev(T); oof = np.load(D / "oof_cv_gte_e8_dup3_int8.npy"); re = np.zeros(len(dev))
for k, (tr_i, te_i) in enumerate(kfold(dev, 5, seed=0)):
    re[te_i] = onnx_probs(D / f"models/cv_gte_e8_dup3/fold{k}", [dev[i] for i in te_i])
print("dev oof re-predict max|diff|", float(np.abs(re - oof).max()))
y = np.array([e.labels["answer"] for e in dev]).astype(int)
print("oof acc@0.5", acc_f1(y, oof)[0], "pos rate", y.mean())
pf = onnx_probs(D / "models/final_gte_e8_dup3", ev)
print("test full re-predict (same batching as frozen) max|diff|", float(np.abs(pf - frozen).max()))
p1 = np.array([onnx_probs(D / "models/final_gte_e8_dup3", [ev[i]], batch=1)[0] for i in idx])
print("batch=1 vs frozen max|diff|", float(np.abs(p1 - frozen[idx]).max()))
