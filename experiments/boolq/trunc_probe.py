import sys, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
from common import *
from ownyourjev.spec import render_state
d = Path(sys.argv[1]); onnx = sys.argv[2]
tok = AutoTokenizer.from_pretrained(d / "backbone")
o = ort.SessionOptions(); o.intra_op_num_threads = 8
s = ort.InferenceSession(str(d / onnx), o, providers=["CPUExecutionProvider"])
dev = protocol.dev(TASK); g = np.array([e.gold[QID] for e in dev]); texts = [render_state(e.state) for e in dev]
lens = np.array([len(tok(t)["input_ids"]) for t in texts]); print("tok lens p50/95/max", np.percentile(lens, [50, 95]), lens.max())
for side in ["right", "left"]:
    tok.truncation_side = side
    for L in [384, 256, 224, 192]:
        p = np.array([s.run(None, {k: v.astype(np.int64) for k, v in tok([t], truncation=True, max_length=L, return_tensors="np").items() if k in ("input_ids", "attention_mask")})[0].reshape(-1)[0] for t in texts])
        print(side, L, "acc", np.mean((p >= .5) == g), "n_trunc", int((lens > L).sum()))
