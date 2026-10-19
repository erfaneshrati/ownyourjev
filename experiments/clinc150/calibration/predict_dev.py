"""Out-of-sample dev predictions with the exact deployed inference path (predict_eval.py):
ONNX (mlp.Wi-only int8) -> OnnxPredictor probs (batch 64, threads 8) -> apply_threshold(tau=0.4).
The deployed model never trained on dev (dev was used only for selection), so these are out-of-sample.
Verifies by re-predicting the first 256 eval_inputs and comparing with the frozen eval_probs.npy."""
import json
from pathlib import Path
import numpy as np
from ownyourjev.export import OnnxPredictor
from benchmarks.v2 import protocol as P
from experiments.clinc150.common import *

d = Path("data/v2/clinc150/models/E_lr5_q_only_Wi"); tau = 0.4
frozen = np.load("data/v2/clinc150/eval_probs.npy")
ev = P.eval_inputs(TASK)
for f in ["model.onnx", "model.int8.onnx"]:
    pr = OnnxPredictor(d, d / f, threads=8)
    def run(exs): return np.concatenate([pr.probs([e.state for e in exs[i:i + 64]]) for i in range(0, len(exs), 64)])
    pe = apply_threshold(run(ev[:256]), tau).astype(np.float32)
    print(f, "max abs diff vs frozen (256 test inputs):", float(np.abs(pe - frozen[:256]).max()),
          "argmax agree", float((pe.argmax(1) == frozen[:256].argmax(1)).mean()))
    if f == "model.int8.onnx":
        dev = P.dev(TASK)
        raw = run(dev)
        pd = apply_threshold(raw, tau).astype(np.float32)
        out = P.RESULTS / TASK
        np.save(out / "dev_probs_deployed.npy", pd)
        np.save("data/v2/clinc150/dev_probs_deployed_prethreshold.npy", raw.astype(np.float32))
        gd = [e.labels["answer"] for e in dev]
        print("dev", metrics(pd, gd))
