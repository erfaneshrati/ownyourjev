"""Export a PAWS model to ONNX (fp32 + int8), dev acc for each, plus CPU latency via benchmarks.v2.latency."""
import sys, subprocess, shutil
from common import *
T = "paws"
d = ROOT / "data/v2/paws/models" / sys.argv[1]
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
if not (d / "model.int8.onnx").is_file():
    export_int8(d)
pt = np.load(d / "dev_probs.npy")
for nm in ["model.onnx", "model.int8.onnx"]:
    p = onnx_probs(d, dev, onnx_path=d / nm)
    np.save(d / f"dev_probs_{nm}.npy", p)
    print(nm, "dev acc", acc_f1(y, p)[0], "torch", acc_f1(y, pt)[0], flush=True)
