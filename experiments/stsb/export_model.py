"""Export a saved model dir to ONNX + int8 and check dev Spearman (plain and with swap-TTA) via onnxruntime."""
import sys
from common import *
from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor
md = Path(sys.argv[1])
if not (md / "model.onnx").exists():
    export_onnx(md)
if not (md / "model.int8.onnx").exists():
    quantize_int8(md / "model.onnx")
dev = protocol.dev(T); g = np.array([e.gold["answer"] for e in dev])
sw = [{"sentence_1": e.state["sentence_2"], "sentence_2": e.state["sentence_1"]} for e in dev]
for name in ["model.onnx", "model.int8.onnx"]:
    p = OnnxPredictor(md, md / name, threads=4)
    a = np.concatenate([p.probs([e.state]) for e in dev]); b = np.concatenate([p.probs([s]) for s in sw])
    np.save(md / f"dev_probs_{name}.npy", a); np.save(md / f"dev_probs_tta_{name}.npy", (a + b) / 2)
    print(name, "dev spearman", round(sp(g, a @ np.arange(6)), 4), "tta", round(sp(g, ((a + b) / 2) @ np.arange(6)), 4))
