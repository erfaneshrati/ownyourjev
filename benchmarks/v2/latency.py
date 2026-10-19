"""Per-request CPU latency of a frozen pipeline (all of a task's question heads, as Jev answers them in one call).

Usage: uv run python -m benchmarks.v2.latency <task> <model_dir> [<model_dir> ...] --out latency.json [--onnx]
Batch-1 end-to-end latency (tokenisation + forward + softmax for every head, sequentially) on the dev texts,
on CPU with 4 threads. The gate is p95 < 70 ms (Jev's best published end-to-end latency).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import typer
from transformers import AutoTokenizer

from benchmarks.v2 import protocol
from ownyourjev.models.decision import DecisionModel
from ownyourjev.spec import render_state

app = typer.Typer()


def _run(models, texts, device, n):
    times = []
    with torch.no_grad():
        for i in range(n):
            t = texts[i % len(texts)]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m, tok in models:
                enc = tok([t], truncation=True, max_length=m.max_len, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    m.probs(m(enc["input_ids"].to(device), enc["attention_mask"].to(device)))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return times[3:] if len(times) > 6 else times


@app.command()
def main(task: str, model_dirs: list[Path], out: Path = typer.Option(...), n: int = 100, threads: int = 4, onnx: bool = typer.Option(False, help="use ONNX Runtime (export first with `ownyourjev export`)")):
    texts = [render_state(e.state) for e in protocol.dev(task)]
    res = {"task": task, "model_dirs": [str(d) for d in model_dirs], "device": "cpu", "threads": threads, "runtime": "onnxruntime" if onnx else "torch"}
    torch.set_num_threads(threads)
    if onnx:
        from ownyourjev.export import OnnxPredictor

        preds = [OnnxPredictor(d, threads=threads) for d in model_dirs]
        times = []
        for i in range(n + 3):
            t0 = time.perf_counter()
            for p in preds:
                p.probs([texts[i % len(texts)]])
            times.append((time.perf_counter() - t0) * 1000)
        c = times[3:]
    else:
        models = [(DecisionModel.load(d, device="cpu"), AutoTokenizer.from_pretrained(Path(d) / "backbone")) for d in model_dirs]
        res["params_m"] = [round(sum(p.numel() for p in m.parameters()) / 1e6, 1) for m, _ in models]
        c = _run(models, texts, "cpu", n + 3)
    res.update(cpu_p50_ms=float(np.percentile(c, 50)), cpu_p95_ms=float(np.percentile(c, 95)), n_requests=len(c))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    app()
