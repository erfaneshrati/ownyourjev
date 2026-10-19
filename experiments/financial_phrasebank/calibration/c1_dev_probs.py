"""Out-of-sample dev predictions of the deployed financial_phrasebank model (dev uids dropped from training in
train_student.py / v1_silver). Same artifact (model.int8.onnx) and same inference path as finalize_probs.py
(OnnxPredictor threads=8, batches of 64). Verifies against the frozen test probs, checks text overlap dev vs train pool."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ag_news"))
from oyj_common import *  # noqa
import numpy as np
from ownyourjev.export import OnnxPredictor
T = "financial_phrasebank"
md = ROOT / "data/v2/financial_phrasebank/models/v1seeded_s1"
R = ROOT / "benchmarks/results_v2" / T
pred = OnnxPredictor(md, threads=8)
print("onnx", pred.sess._model_path if hasattr(pred.sess, "_model_path") else "(default: model.int8.onnx)")
def run(exs):
    return np.concatenate([pred.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])
q = protocol.task_spec(T).parsed["answer"]
dev = protocol.dev(T); y = np.array([q.options.index(e.labels["answer"]) for e in dev])
p = run(dev)
print("dev n", len(dev), "acc", float((p.argmax(1) == y).mean()))
for name in ["benchmarks/results_v2/financial_phrasebank/dev_probs_deployed.npy", "data/v2/financial_phrasebank/models/v1seeded_s1/dev_probs_int8.npy"]:
    o = np.load(ROOT / name); print("max |diff| vs", name, float(np.abs(p - o).max()))
ev = protocol.eval_inputs(T); frozen = np.load(R / "eval_probs.npy")
n = 256
pt = run(ev[:n])
print(f"test first {n}: max |diff| vs frozen", float(np.abs(pt - frozen[:n]).max()), "argmax agree", float((pt.argmax(1) == frozen[:n].argmax(1)).mean()))
# training-pool text overlap with dev (exact text after normalisation)
silver = v1_silver(T, "silver_seeded")
norm = lambda s: " ".join(json.dumps(s, sort_keys=True).lower().split())
pool_txt = {norm(e.state) for e in silver.values()}
ov = sum(norm(e.state) in pool_txt for e in dev)
print("train pool size", len(silver), "dev texts also in train pool:", ov)
np.save(ROOT / "experiments/financial_phrasebank/calibration/dev_probs_c1.npy", p)
