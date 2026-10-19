"""Predict the 200 dev examples with the frozen deployed int8 ONNX (same path as predict_eval.py) and verify it
reproduces the frozen test probabilities on a few eval inputs. Also checks dev texts are not training texts."""
import json, numpy as np
from pathlib import Path
from benchmarks.v2 import protocol as P
from ownyourjev.export import OnnxPredictor

T = "banking77"
d = Path("data/v2/banking77/models/votesoft")
p = OnnxPredictor(d, threads=4)  # default onnx = model.int8.onnx, as in predict_eval.py

# leakage check: training set = pool + seeds (train_student.build)
dev = P.dev(T)
train = P.pool(T) + P.seeds(T)
tr_uid = {e.uid for e in train}
norm = lambda s: " ".join(str(s).lower().split())
tr_txt = {norm(e.state) for e in train}
uid_overlap = sum(e.uid in tr_uid for e in dev)
txt_overlap = [e.uid for e in dev if norm(e.state) in tr_txt]
print("dev uid in train:", uid_overlap, "| dev text (normalised) in train:", len(txt_overlap))

# reproducibility check on eval inputs, batches of 64 as predict_eval.py
ev = P.eval_inputs(T)
frozen = np.load("benchmarks/results_v2/banking77/eval_probs.npy")
idx = list(range(0, 128)) + list(range(len(ev) - 64, len(ev)))
re = np.concatenate([p.probs([e.state for e in ev[0:64]]), p.probs([e.state for e in ev[64:128]]),
                     p.probs([e.state for e in ev[len(ev) - 64:]])]) if len(ev) % 64 == 0 else None
re = np.concatenate([p.probs([e.state for e in ev[0:64]]), p.probs([e.state for e in ev[64:128]])])
diff = float(np.abs(re - frozen[:128]).max())
print("max |re-predicted - frozen| on 128 eval inputs:", diff)
assert diff < 1e-4

pd = np.concatenate([p.probs([e.state for e in dev[i:i + 64]]) for i in range(0, len(dev), 64)])
q = P.task_spec(T).parsed["answer"]
y = np.array([q.options.index(e.labels["answer"]) for e in dev])
np.save("benchmarks/results_v2/banking77/dev_probs_deployed.npy", pd)
np.save("experiments/banking77/calibration/dev_y.npy", y)
print("dev acc int8:", float((pd.argmax(1) == y).mean()), pd.shape)
json.dump({"uid_overlap": uid_overlap, "text_overlap": txt_overlap, "repro_maxdiff": diff},
          open("experiments/banking77/calibration/checks.json", "w"), indent=1)
