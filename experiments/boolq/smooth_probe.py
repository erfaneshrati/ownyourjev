import sys, numpy as np
from common import *
from fast_export import lean_export
from predict_onnx import run
from ownyourjev.export import OnnxPredictor
d = Path(sys.argv[1])
dev = protocol.dev(TASK); g = np.array([e.gold[QID] for e in dev])
pf = np.load(d / "dev_probs_model.npy")
for a in [float(x) for x in sys.argv[2:]]:
    lean_export(d, d / f"sm{a}.onnx", check=(a == float(sys.argv[2])), smooth=a)
    for nm in [f"sm{a}.onnx", f"sm{a}.int8.onnx"]:
        p = run(OnnxPredictor(d, d / nm, threads=8), dev)
        print(f"{nm:16s} acc={np.mean((p>=.5)==g):.3f} agree_fp32={np.mean((p>=.5)==(pf>=.5)):.3f} mean|dp|={np.abs(p-pf).mean():.3f}", flush=True)
