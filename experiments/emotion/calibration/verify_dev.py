"""Verify that benchmarks/results_v2/emotion/dev_probs_deployed.npy comes from the deployed artifact.
It was written by experiments/ag_news/finalize_probs.py in the same run as the frozen eval_probs.npy
(OnnxPredictor(data/v2/emotion/models/g1_10k_fp32, threads=8), batches of 64, e.state inputs).
Re-predicts the first 128 dev and 128 eval inputs with that exact path and compares."""
import json
import numpy as np
from ownyourjev.export import OnnxPredictor
from benchmarks.v2 import protocol as P

T, md = "emotion", "data/v2/emotion/models/g1_10k_fp32"
pred = OnnxPredictor(md, threads=8)
print("onnx file:", getattr(pred.sess, "_model_path", None))
def run(exs):
    return np.concatenate([pred.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])
pd_saved = np.load(P.RESULTS / T / "dev_probs_deployed.npy")
pe_frozen = np.load(json.loads((P.RESULTS / T / "FROZEN.json").read_text())["heads"]["answer"]["probs"])
dev, ev = P.dev(T), P.eval_inputs(T)
pd = run(dev[:128]); pe = run(ev[:128])
r = {"dev_shape": list(pd_saved.shape), "eval_shape": list(pe_frozen.shape),
     "dev_maxabs_128": float(np.abs(pd - pd_saved[:128]).max()),
     "eval_maxabs_128": float(np.abs(pe - pe_frozen[:128]).max()),
     "dev_argmax_agree": float((pd.argmax(1) == pd_saved[:128].argmax(1)).mean()),
     "eval_argmax_agree": float((pe.argmax(1) == pe_frozen[:128].argmax(1)).mean())}
q = P.task_spec(T).parsed["answer"]
y = np.array([q.options.index(e.labels["answer"]) for e in dev])
r["dev_acc_saved"] = float((pd_saved.argmax(1) == y).mean())
print(json.dumps(r, indent=1))
