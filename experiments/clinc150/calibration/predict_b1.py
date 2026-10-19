"""Batch-size-1 predictions with the deployed artifact (model.int8.onnx, mlp.Wi-only int8), threads=8 as in
predict_eval.py, one text per call, then apply_threshold(tau=0.4) (the deployed oos rule, baked into the vector
exactly as in the frozen eval_probs.npy). Also saves pre-threshold vectors."""
import time
from pathlib import Path
import numpy as np
from ownyourjev.export import OnnxPredictor
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

d = Path("data/v2/clinc150/models/E_lr5_q_only_Wi"); tau = 0.4; out = Path("data/v2/clinc150")
pr = OnnxPredictor(d, d / "model.int8.onnx", threads=8)
def run(exs): return np.concatenate([pr.probs([e.state]) for e in exs])
t = time.time()
dev = P.dev(TASK); rd = run(dev)
np.save(out / "dev_probs_b1_prethreshold.npy", rd.astype(np.float32))
pd = apply_threshold(rd, tau).astype(np.float32); np.save(out / "dev_probs_b1.npy", pd)
print("dev b1", metrics(pd, [e.labels["answer"] for e in dev]), "max|dp| vs batched", float(np.abs(pd - np.load(P.RESULTS / TASK / "dev_probs_deployed.npy")).max()), flush=True)
re = run(P.eval_inputs(TASK))
np.save(out / "eval_probs_b1_prethreshold.npy", re.astype(np.float32))
pe = apply_threshold(re, tau).astype(np.float32); np.save(out / "eval_probs_b1.npy", pe)
f = np.load(out / "eval_probs.npy")
print("eval b1 done", time.time() - t, "s; decision flips (thresholded argmax) vs frozen:", int((pe.argmax(1) != f.argmax(1)).sum()),
      "/", len(pe), "max|dp|", float(np.abs(pe - f).max()), "pred oos share", float((pe.argmax(1) == OOS).mean()))
