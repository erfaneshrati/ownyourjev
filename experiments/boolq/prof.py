import sys, json, glob, os, collections, numpy as np, onnxruntime as ort
p, L = sys.argv[1], int(sys.argv[2])
o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.enable_profiling = True
o.profile_file_prefix = "/tmp/claude-1000/-media-erfan-Data-ownyourjev/887ffe38-400c-4edb-bb16-1a88de00b3d4/scratchpad/prof"
s = ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
ids = np.random.randint(1000, 20000, size=(1, L)).astype(np.int64)
for _ in range(10): s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
f = s.end_profiling(); ev = json.load(open(f))
agg = collections.Counter()
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        agg[e["args"]["op_name"]] += e["dur"]
tot = sum(agg.values())
for k, v in agg.most_common(12): print(f"{k:28s} {v/10/1000:7.1f} ms  {100*v/tot:.0f}%")
os.remove(f)
