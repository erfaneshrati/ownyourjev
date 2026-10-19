"""Per-request CPU latency of the multi-head pipeline, mirroring benchmarks.v2.latency --onnx: batch 1, 4 threads,
dev states in order, 3 warm-up requests then n timed; each request = tokenise + truncate + ONNX forward + softmax (all 5 heads)
[+ h/c blend post-processing if post.json says so]. Also reports dev Spearman of the ONNX model.
Usage: latency_mh.py <model_dir> --out latency.json [--onnx-name model.int8.onnx] [--n 100]"""
import argparse, json, time
from pathlib import Path
import numpy as np
from common import *
import mh
ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--out"); ap.add_argument("--onnx_name", default="model.int8.onnx")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--eval", action="store_true")
a = ap.parse_args()
p = mh.OnnxMH(a.model_dir, a.onnx_name, threads=a.threads)
post = json.loads((Path(a.model_dir) / "post.json").read_text()) if (Path(a.model_dir) / "post.json").exists() else {}
from postproc import blend_hc
dev = protocol.dev(T)
states = [e.state for e in dev]
def request(s):
    P = p.probs(s)
    if post.get("blend_hc"):
        P = blend_hc(P)
    return P
times = []
for i in range(a.n + 3):
    t0 = time.perf_counter(); request(states[i % len(states)]); times.append((time.perf_counter() - t0) * 1000)
c = times[3:]
res = {"task": T, "model_dirs": [a.model_dir], "onnx": a.onnx_name, "device": "cpu", "threads": a.threads, "runtime": "onnxruntime",
       "pipeline": "experiments/helpsteer2/latency_mh.py (tokenise+truncate+forward+softmax, 5 heads, one encoder)",
       "cpu_p50_ms": float(np.percentile(c, 50)), "cpu_p95_ms": float(np.percentile(c, 95)), "n_requests": len(c)}
if a.eval:
    P = np.stack([request(s) for s in states]); es = P @ np.arange(5)
    rs = [sp(es[:, j], [e.gold[x] for e in dev]) for j, x in enumerate(A)]
    res["dev_spearman"] = dict(zip(A, rs)); res["dev_mean"] = float(np.mean(rs))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
