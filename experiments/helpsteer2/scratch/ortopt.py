import sys, time; sys.path.insert(0, "/media/erfan/Data/ownyourjev/experiments/helpsteer2")
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, collections
from onnxruntime.transformers import optimizer
d = Path(sys.argv[1])
m = optimizer.optimize_model(str(d / "model.onnx"), model_type="bert", num_heads=12, hidden_size=768, opt_level=1)
print(m.get_fused_operator_statistics())
m.save_model_to_file(str(d / "model.opt.onnx"))
g0 = onnx.load(str(d / "model.onnx"), load_external_data=False); g1 = onnx.load(str(d / "model.opt.onnx"), load_external_data=False)
print("nodes", len(g0.graph.node), "->", len(g1.graph.node))
print(collections.Counter(n.op_type for n in g1.graph.node).most_common(12))
