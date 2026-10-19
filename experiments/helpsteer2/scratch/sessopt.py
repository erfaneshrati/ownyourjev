import sys, time; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
import numpy as np, onnxruntime as ort
from common import *
import mh
d = sys.argv[1]
base = mh.OnnxMH(d, "model.int8.onnx", threads=4)
def mk(cfg):
    o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.inter_op_num_threads = 1
    for k, v in cfg.items(): o.add_session_config_entry(k, v)
    return ort.InferenceSession(f"{d}/model.int8.onnx", o, providers=["CPUExecutionProvider"])
variants = {"default": mk({}), "dbb4": mk({"session.dynamic_block_base": "4"}), "dbb2": mk({"session.dynamic_block_base": "2"}), "daz": mk({"session.set_denormal_as_zero": "1"})}
st = [e.state for e in protocol.dev(T)]
ids_all = [np.asarray([base.enc.ids(s)], dtype=np.int64) for s in st]
R = {k: [] for k in variants}
for i in range(203):
    for k, s in variants.items():
        ids = ids_all[i % 200]
        t0 = time.perf_counter(); s.run(["probs"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}); R[k].append((time.perf_counter() - t0) * 1000)
for k, t in R.items(): print(k, "p50 %.1f p95 %.1f" % (np.percentile(t[3:], 50), np.percentile(t[3:], 95)))
