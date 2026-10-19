import sys, json; sys.path.insert(0,"experiments/sst5")
import numpy as np
from lab import DEV, DEVY, P, T, report
from ownyourjev.export import OnnxPredictor
d = "data/v2/sst5/models/final"; pr = OnnxPredictor(d, threads=8)  # picks model.int8.onnx (partial int8)
def run(exs): return np.concatenate([pr.probs([e.state for e in exs[i:i+32]]) for i in range(0, len(exs), 32)])
pd = run(DEV); report("final-dev", pd)
ev = P.eval_inputs(T); pe = run(ev)
out = "benchmarks/results_v2/sst5"; np.save(f"{out}/dev_probs.npy", pd); np.save(f"{out}/eval_probs.npy", pe)
print(pe.shape, np.bincount(pe.argmax(1), minlength=5))
