"""ONNX export of a DecisionModel (backbone + pooling + head + temperature) and an onnxruntime predictor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ownyourjev.models.decision import DecisionModel, load_calibrator
from ownyourjev.spec import Question, render_state


class _Wrapper(nn.Module):
    def __init__(self, m: DecisionModel):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        logits = self.m(input_ids, attention_mask)
        return self.m.probs(logits)


DEFAULT_FP32_PATTERNS = ("mlp.Wo", "mlp/Wo", "down_proj")  # MLP output projections: activation outliers live here


def quantize_int8(
    onnx_path: str | Path,
    out_path: str | Path | None = None,
    *,
    per_channel: bool = True,
    keep_fp32: tuple[str, ...] | None = DEFAULT_FP32_PATTERNS,
) -> Path:
    """Dynamic int8 weight quantization for CPU inference (MatMul/Gemm only).

    ``keep_fp32``: node-name substrings left in fp32. Excluding the MLP output projections (default) keeps
    fp32 accuracy at near-int8 speed; four independent experiments found that quantising them is what
    costs accuracy (UltraBERT collapses without this). Pass ``()`` to quantise everything. Always check
    dev accuracy afterwards."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    out_path = Path(out_path) if out_path else onnx_path.with_suffix(".int8.onnx")
    exclude: list[str] = []
    if keep_fp32:
        g = onnx.load(str(onnx_path), load_external_data=False).graph
        exclude = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm") and any(p in n.name for p in keep_fp32)]
    quantize_dynamic(
        str(onnx_path), str(out_path), weight_type=QuantType.QInt8, per_channel=per_channel,
        op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=exclude,
    )
    return out_path


def export_onnx(model_dir: str | Path, out_path: str | Path | None = None, *, opset: int = 17) -> Path:
    model_dir = Path(model_dir)
    out_path = Path(out_path) if out_path else model_dir / "model.onnx"
    m = DecisionModel.load(model_dir, device="cpu").float().eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
    enc = tok(["example input for tracing", "second"], padding=True, return_tensors="pt")
    torch.onnx.export(
        _Wrapper(m),
        (enc["input_ids"], enc["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["probs"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "probs": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return out_path


class OnnxPredictor:
    def __init__(self, model_dir: str | Path, onnx_path: str | Path | None = None, threads: int | None = None):
        model_dir = Path(model_dir)
        if onnx_path is None:  # prefer the int8 model when it exists
            q8 = model_dir / "model.int8.onnx"
            onnx_path = q8 if q8.is_file() else model_dir / "model.onnx"
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        self.question = Question(**meta["question"])
        self.max_len = meta["max_len"]
        self.tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path or model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._dummy = DecisionModel.__new__(DecisionModel)  # only for .answer()
        self._dummy.question = self.question
        self._dummy.calibrator = load_calibrator(model_dir)

    def probs(self, states: list) -> np.ndarray:
        enc = self.tok([render_state(s) for s in states], padding=True, truncation=True, max_length=self.max_len, return_tensors="np")
        out = self.sess.run(["probs"], {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        return out

    def answer(self, state):
        return self._dummy.answer(torch.as_tensor(self.probs([state])[0]))

    def benchmark_latency(self, state, n: int = 50) -> dict:
        self.probs([state])
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.probs([state])
            t.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(np.mean(t))}
