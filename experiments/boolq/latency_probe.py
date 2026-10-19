"""Probe CPU latency of untrained gte-modernbert (optionally with fewer layers) as int8 ONNX on boolq dev texts."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
from benchmarks.v2 import protocol
from ownyourjev.spec import render_state
from ownyourjev.models.backbones import resolve_backbone
from ownyourjev.models.decision import DecisionModel, load_backbone
from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor

OUT = Path("data/v2/boolq/models/probe")
spec = resolve_backbone("gte-modernbert")
q = protocol.task_spec("boolq").parsed["answer"]
dev = protocol.dev("boolq")
texts = [render_state(e.state) for e in dev]
tok = AutoTokenizer.from_pretrained(spec.hf_id)
lens = np.array([len(tok(t)["input_ids"]) for t in texts])
print("dev token lens p50/p90/p95/p99/max", np.percentile(lens, [50, 90, 95, 99]), lens.max())
for nl in [int(x) for x in sys.argv[1:]] or [22]:
    bb, tok = load_backbone(spec)
    if nl < bb.config.num_hidden_layers:
        bb.layers = bb.layers[:nl]; bb.config.layer_types = bb.config.layer_types[:nl]; bb.config.num_hidden_layers = nl
    m = DecisionModel(q, spec, bb, max_len=384)
    d = OUT / f"L{nl}"; m.save(d, tok)
    import sys as _s; _s.path.insert(0,'experiments/boolq'); from fast_export import lean_export; lean_export(d, check=False)
    for name in ["model.int8.onnx"]:
        p = OnnxPredictor(d, d / name, threads=4)
        ts = []
        for i in range(103):
            t0 = time.perf_counter(); p.probs([dev[i % len(dev)].state]); ts.append((time.perf_counter() - t0) * 1000)
        ts = ts[3:]
        print(f"layers={nl} {name}: p50={np.percentile(ts,50):.1f} p95={np.percentile(ts,95):.1f}", flush=True)
