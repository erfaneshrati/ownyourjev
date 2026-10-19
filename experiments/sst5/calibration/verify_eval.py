"""Re-verify against frozen eval_probs.npy using predict_final.py's exact batching (contiguous batches of 32 in
eval order). Dynamic int8 activation scales are per batch tensor, so a random 64-subset (batched differently)
does not reproduce bit-for-bit; same-batch composition should."""
import json
import numpy as np
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor
pr = OnnxPredictor("data/v2/sst5/models/final", threads=8)
ev = P.eval_inputs("sst5"); fe = np.load("benchmarks/results_v2/sst5/eval_probs.npy")
res = {}
for s in (0, 32, 1024, 2176):
    pe = pr.probs([e.state for e in ev[s:s+32]]); res[s] = float(np.abs(pe - fe[s:s+32]).max())
    print("batch", s, "max diff", res[s], "argmax agree", (pe.argmax(1) == fe[s:s+32].argmax(1)).mean())
pe = np.concatenate([pr.probs([e.state]) for e in ev[:32]])
print("batch-1 on first 32: max diff", np.abs(pe - fe[:32]).max(), "argmax agree", (pe.argmax(1) == fe[:32].argmax(1)).mean())
v = json.load(open("data/v2/sst5/calib_verify.json")); v["same_batching_max_diff"] = res
json.dump(v, open("data/v2/sst5/calib_verify.json", "w"))
