"""Predict with the deployed artifact (ONNX int8 multi-head + optional h/c blend) on dev or eval_inputs.
Usage: predict.py <model_dir> <dev|eval> [--onnx_name ...] [--blend_hc] -> <model_dir>/<split>_probs_final.npy and per-head files."""
import argparse, json
from pathlib import Path
import numpy as np
from common import *
import mh
ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("split"); ap.add_argument("--onnx_name", default="model.int8.onnx")
a = ap.parse_args()
d = Path(a.model_dir)
exs = protocol.dev(T) if a.split == "dev" else protocol.eval_inputs(T)
p = mh.OnnxMH(d, a.onnx_name, threads=8)
P = np.stack([p.probs(e.state) for e in exs])
post = json.loads((d / "post.json").read_text()) if (d / "post.json").exists() else {}
if post.get("blend_hc"):
    from postproc import blend_hc
    P = blend_hc(P)
np.save(d / f"{a.split}_probs_final.npy", P)
for j, x in enumerate(A):
    np.save(d / f"{a.split}_{x}.npy", P[:, j])
if a.split == "dev":
    es = P @ np.arange(5); rs = [sp(es[:, j], [e.gold[x] for e in exs]) for j, x in enumerate(A)]
    print("dev", dict(zip(A, [round(r, 4) for r in rs])), "mean", round(float(np.mean(rs)), 4))
print(P.shape)
