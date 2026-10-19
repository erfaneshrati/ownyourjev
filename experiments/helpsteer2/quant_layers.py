"""Which layers' mlp.Wo can be int8 without hurting? Greedy: quantise mlp.Wo in each layer alone (rest fp32 Wo), measure
agreement (corr of expected scores, 5 flash heads) with the fp32 model on dev."""
import sys, json
from pathlib import Path
import numpy as np, onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from common import *
import mh
d = Path(sys.argv[1]); dev = protocol.dev(T)[:100]
g = onnx.load(str(d / "model.onnx"), load_external_data=False)
mm = [n.name for n in g.graph.node if n.op_type in ("MatMul", "Gemm")]
def run(name):
    p = mh.OnnxMH(d, name, threads=8); return np.stack([p.probs(e.state) for e in dev])[:, :5] @ np.arange(5)
ref = run("model.onnx")
nl = json.loads((d / "meta.json").read_text())["n_layers"]
res = {}
for L in range(nl):
    excl = [n for n in mm if "heads" in n or ("mlp/Wo" in n and f"layers.{L}/" not in n)]
    out = d / "tmp_q.onnx"
    quantize_dynamic(str(d / "model.onnx"), str(out), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    e = run("tmp_q.onnx"); ag = float(np.mean([np.corrcoef(e[:, j], ref[:, j])[0, 1] for j in range(5)]))
    res[L] = ag; print(L, f"{ag:.4f}", flush=True)
(d / "tmp_q.onnx").unlink()
(d / "quant_layers.json").write_text(json.dumps(res))
