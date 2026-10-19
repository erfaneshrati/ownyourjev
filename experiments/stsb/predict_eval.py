"""Eval-input probs from the deployed ONNX model (batch 1, as served)."""
import sys
from common import *
from ownyourjev.export import OnnxPredictor
md = Path(sys.argv[1]); out = Path(sys.argv[2])
p = OnnxPredictor(md, threads=4)
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
a = np.concatenate([p.probs([e.state]) for e in dev]); print("dev spearman (deployed onnx)", round(sp(g, a @ np.arange(6)), 4))
ev = protocol.eval_inputs(T)
pe = np.concatenate([p.probs([e.state]) for e in ev]).astype(np.float64)
out.parent.mkdir(parents=True, exist_ok=True); np.save(out, pe); np.save(out.with_name("dev_probs.npy"), a)
print(pe.shape, pe.sum(1).min(), pe.sum(1).max())
