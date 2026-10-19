"""Calibration study: out-of-sample dev predictions of the frozen sst5 model (same path as predict_final.py),
plus verification against the frozen test probabilities (eval_probs.npy) on a subset of eval inputs."""
import json
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor

T = "sst5"; OUT = "benchmarks/results_v2/sst5"
d = "data/v2/sst5/models/final"; pr = OnnxPredictor(d, threads=8)  # picks model.int8.onnx, as predict_final.py
print("onnx:", getattr(pr, "onnx_path", None) or getattr(pr, "path", None))
def run(exs): return np.concatenate([pr.probs([e.state for e in exs[i:i+32]]) for i in range(0, len(exs), 32)])
dev = P.dev(T)
pd = run(dev)
y = np.array([int(min(max(round(float(e.gold["answer"])), 0), 4)) for e in dev])
print("dev acc", (pd.argmax(1) == y).mean())
old = np.load(f"{OUT}/dev_probs.npy")
print("max |dev - frozen dev_probs.npy|", np.abs(pd - old).max())
ev = P.eval_inputs(T); idx = np.random.default_rng(0).choice(len(ev), 64, replace=False)
pe = run([ev[i] for i in idx]); fe = np.load(f"{OUT}/eval_probs.npy")[idx]
diff = float(np.abs(pe - fe).max()); print("max |eval - frozen eval_probs| on 64 test inputs", diff)
np.save(f"{OUT}/dev_probs_deployed.npy", pd); np.save("data/v2/sst5/calib_dev_y.npy", y)
json.dump({"dev_acc": float((pd.argmax(1) == y).mean()), "max_diff_dev_vs_frozen_dev": float(np.abs(pd - old).max()),
           "max_diff_eval_64": diff, "eval_idx": idx.tolist()}, open("data/v2/sst5/calib_verify.json", "w"))
