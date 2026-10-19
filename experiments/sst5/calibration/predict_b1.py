"""Batch-1 (serving mode) dev + test predictions of the frozen sst5 model: same artifact/path as predict_final.py, one text per call."""
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor
OUT = "benchmarks/results_v2/sst5"
pr = OnnxPredictor("data/v2/sst5/models/final", threads=8)
def run(exs): return np.concatenate([pr.probs([e.state]) for e in exs])
pd = run(P.dev("sst5")); np.save(f"{OUT}/dev_probs_b1.npy", pd); print("dev done", pd.shape, flush=True)
pe = run(P.eval_inputs("sst5")); np.save(f"{OUT}/eval_probs_b1.npy", pe); print("eval done", pe.shape)
