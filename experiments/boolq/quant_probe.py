import sys, numpy as np
from common import *
from quant import quantize_weights_only
from predict_onnx import run
from ownyourjev.export import OnnxPredictor
d = Path(sys.argv[1])
dev = protocol.dev(TASK); g = np.array([e.gold[QID] for e in dev])
pf = np.load(d / "dev_probs_model.npy")
for tag, pat in [(x.split("=")[0], x.split("=")[1] or None) for x in sys.argv[2:]]:
    nodes = quantize_weights_only(d / "model.onnx", d / f"q_{tag}.onnx", pat)
    p = run(OnnxPredictor(d, d / f"q_{tag}.onnx", threads=8), dev)
    print(f"{tag:12s} nq={len(nodes)} acc={np.mean((p>=.5)==g):.3f} agree_fp32={np.mean((p>=.5)==(pf>=.5)):.3f} mean|dp|={np.abs(p-pf).mean():.3f}", flush=True)
