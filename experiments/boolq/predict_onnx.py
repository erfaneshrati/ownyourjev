"""Lean-export a trained model to ONNX (fp32 + int8) and score dev (and optionally eval_inputs) with onnxruntime."""
import argparse, json, sys
import numpy as np
from common import *
from fast_export import lean_export
from ownyourjev.export import OnnxPredictor
from ownyourjev.evaluate import binary_metrics


def run(pred, exs, bs=16):
    order = sorted(range(len(exs)), key=lambda i: len(exs[i].state["passage"]) + len(exs[i].state["question"]))
    out = np.zeros(len(exs), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        out[idx] = np.asarray(pred.probs([exs[i].state for i in idx])).reshape(-1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--smooth", type=float, default=0.8); ap.add_argument("--eval", action="store_true"); ap.add_argument("--no_export", action="store_true")
    a = ap.parse_args()
    d = Path(a.model_dir)
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(d / "backbone")
    if t.truncation_side != "left":  # the question comes last in the rendered state: truncate the passage start
        t.truncation_side = "left"; t.save_pretrained(d / "backbone")
    if not a.no_export:
        lean_export(d, smooth=a.smooth)
    dev = protocol.dev(TASK); g = [e.gold[QID] for e in dev]
    res = {}
    for name in ["model.onnx", "model.int8.onnx"]:
        p = run(OnnxPredictor(d, d / name, threads=8), dev)
        np.save(d / f"dev_probs_{name.replace('.onnx','')}.npy", p)
        m = binary_metrics(g, p)
        res[name] = {"dev_acc": m["accuracy"], "A": float(np.mean((p[:100] >= .5) == np.array(g[:100]))), "B": float(np.mean((p[100:] >= .5) == np.array(g[100:]))), "auroc": m["auroc"]}
        print(name, json.dumps(res[name]), flush=True)
    (d / "onnx_dev.json").write_text(json.dumps(res, indent=1))
    if a.eval:
        ev = protocol.eval_inputs(TASK)
        p = run(OnnxPredictor(d, d / "model.int8.onnx", threads=8), ev)
        np.save(d / "eval_probs_int8.npy", p)
        print("eval probs", p.shape, float((p >= .5).mean()))
