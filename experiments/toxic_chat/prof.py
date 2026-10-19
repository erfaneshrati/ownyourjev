import sys, json, glob, os, collections, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
from common import *
md = D / "models" / sys.argv[1]; f = sys.argv[2]
tok = AutoTokenizer.from_pretrained(md / "backbone")
o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.enable_profiling = True
o.profile_file_prefix = "/tmp/claude-1000/-media-erfan-Data-ownyourjev/887ffe38-400c-4edb-bb16-1a88de00b3d4/scratchpad/prof"
s = ort.InferenceSession(str(md / f), o, providers=["CPUExecutionProvider"])
enc = tok(["word " * 176], return_tensors="np")
feed = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
import time
for _ in range(5): s.run(None, feed)
ts = []
for _ in range(20):
    t = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t) * 1000)
p = s.end_profiling()
ev = json.load(open(p)); agg = collections.Counter()
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        agg[e["args"]["op_name"]] += e["dur"]
tot = sum(agg.values())
print(f, "median ms", np.median(ts), "seq", enc["input_ids"].shape)
for k, v in agg.most_common(12): print(f"  {k:28s} {100*v/tot:5.1f}%")
os.remove(p)
