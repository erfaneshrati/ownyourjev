import time, json, numpy as np, onnxruntime as ort, collections, sys
D="/tmp/claude-1000/-media-erfan-Data-ownyourjev/887ffe38-400c-4edb-bb16-1a88de00b3d4/scratchpad"
p=f"{D}/lat12.int8.onnx"
o=ort.SessionOptions(); o.intra_op_num_threads=4; o.enable_profiling=True
s=ort.InferenceSession(p,o,providers=["CPUExecutionProvider"])
L=int(sys.argv[1])
ids=np.random.randint(1000,5000,(1,L)).astype(np.int64); am=np.ones_like(ids)
for k in range(10): s.run(["probs"],{"input_ids":ids,"attention_mask":am})
f=s.end_profiling()
ev=json.load(open(f)); agg=collections.Counter()
for e in ev:
    if e.get("cat")=="Node": agg[e["args"].get("op_name","?")]+=e["dur"]
tot=sum(agg.values()); print(L, "total ms/run", tot/10/1000)
for k,v in agg.most_common(12): print(k, round(v/10/1000,2))
