"""ONNX export (+ dynamic int8) of a multi-head summeval model, an onnxruntime predictor that includes the full
request path (input rendering, tokenisation, lexical features), and a latency benchmark mirroring
benchmarks.v2.latency (dev texts, batch 1, 4 threads, 100 requests after 3 warm-up).
usage: export_mh.py export <model_dir> | latency <model_dir> --out latency.json | check <model_dir>"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import *
import mh
from feats import features


class _W(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ids, mask, feats): return self.m.probs(self.m(ids, mask, feats))


def export(model_dir):
    model_dir = Path(model_dir)
    m, tok = mh.load(model_dir, "cpu", attn="eager")
    m = m.float().eval()
    from fastmb import FastMH
    fm = FastMH(m, m.cfg.max_len).eval()
    # equivalence check of the export-friendly forward against the reference model
    dev = P.dev(T)[:6]
    texts = [render(e.state, m.cfg.ctx, m.cfg.ctx_chars) for e in dev]
    enc = tok(texts, padding=True, truncation=True, max_length=m.cfg.max_len, return_tensors="pt")
    ff = torch.as_tensor(mh.featurize(dev))
    with torch.no_grad():
        ref = m.probs(m(enc["input_ids"], enc["attention_mask"], ff)); new = fm(enc["input_ids"], enc["attention_mask"], ff)
    print("fast-vs-ref max abs diff", float((ref - new).abs().max()))
    enc = tok(["example input for tracing", "second one"], padding=True, return_tensors="pt")
    f = torch.zeros(2, len(m.f_mu))
    p = model_dir / "model.onnx"
    torch.onnx.export(fm, (enc["input_ids"], enc["attention_mask"], f), str(p), input_names=["input_ids", "attention_mask", "feats"],
                      output_names=["probs"], dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "feats": {0: "b"}, "probs": {0: "b"}},
                      opset_version=17, dynamo=False)
    quantize(model_dir)
    print("exported", p)


def quantize(model_dir):
    """Dynamic int8 (per-channel) for every weight MatMul/Gemm EXCEPT the MLP down-projections (Wo of the GLU, 1152x768):
    their inputs (gelu(a)*b) have outlier activations and per-tensor uint8 activation quantization costs ~0.03 dev.
    Keeping them fp32 recovers fp32 dev accuracy for ~6% more latency."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    model_dir = Path(model_dir); src = model_dir / "model.onnx"
    m = onnx.load(str(src)); init = {i.name: tuple(i.dims) for i in m.graph.initializer}
    excl = [n.name for n in m.graph.node if n.op_type == "MatMul" and init.get(n.input[1]) == (1152, 768)]
    quantize_dynamic(str(src), str(model_dir / "model.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)
    return excl


class Predictor:
    def __init__(self, model_dir, threads=4, int8=True):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "mh_cfg.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        if os.environ.get("ORT_NOSPIN"): o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(model_dir / ("model.int8.onnx" if int8 else "model.onnx")), o, providers=["CPUExecutionProvider"])
        self.use_feats = self.cfg["use_feats"]

    def probs(self, states):
        enc = self.tok([render(s, self.cfg.get("ctx", "lead"), self.cfg.get("ctx_chars", 700)) for s in states], padding=True, truncation=True, max_length=self.cfg["max_len"], return_tensors="np")
        f = np.stack([features(s["summary"], s["source_document"]) for s in states]).astype(np.float32)
        return self.sess.run(["probs"], {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64), "feats": f})[0]


def latency(model_dir, out, n=100, threads=4):
    torch.set_num_threads(threads)
    states = [e.state for e in P.dev(T)]
    pr = Predictor(model_dir, threads)
    times = []
    for i in range(n + 3):
        t0 = time.perf_counter(); pr.probs([states[i % len(states)]]); times.append((time.perf_counter() - t0) * 1000)
    c = times[3:]
    res = {"task": T, "model_dirs": [str(model_dir)], "device": "cpu", "threads": threads, "runtime": "onnxruntime",
           "notes": "experiments/summeval/export_mh.py: same protocol as benchmarks.v2.latency (dev texts, batch 1, 4 threads); one shared encoder answers all 4 heads; timing includes rendering, tokenisation and lexical features",
           "cpu_p50_ms": float(np.percentile(c, 50)), "cpu_p95_ms": float(np.percentile(c, 95)), "n_requests": len(c)}
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))


def check(model_dir, int8=True):
    """dev metric of the ONNX model (int8 or fp32)."""
    dev = P.dev(T); pr = Predictor(model_dir, 8, int8)
    p = np.concatenate([pr.probs([e.state]) for e in dev])
    pred = {d: p[:, j] @ np.arange(5) for j, d in enumerate(DIMS)}
    r = score_dev(pred, dev); print("onnx", "int8" if int8 else "fp32", {k: round(v, 4) for k, v in r.items()}); return p


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "export": export(d)
    elif cmd == "latency": latency(d, sys.argv[sys.argv.index("--out") + 1], threads=int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4)
    elif cmd == "check": check(d, True); check(d, False)


class EnsemblePredictor:
    """Several fold-compatible models (same tokenizer/max_len/ctx) answering one request: shared rendering, tokenisation
    and features; ONNX sessions run one after another; probabilities averaged."""
    def __init__(self, model_dirs, threads=8):
        self.ps = [Predictor(d, threads) for d in model_dirs]
        c0 = self.ps[0].cfg
        assert all(p.cfg["max_len"] == c0["max_len"] and p.cfg.get("ctx_chars", 700) == c0.get("ctx_chars", 700) for p in self.ps)

    def probs(self, states):
        p0 = self.ps[0]
        enc = p0.tok([render(s, p0.cfg.get("ctx", "lead"), p0.cfg.get("ctx_chars", 700)) for s in states], padding=True, truncation=True, max_length=p0.cfg["max_len"], return_tensors="np")
        f = np.stack([features(s["summary"], s["source_document"]) for s in states]).astype(np.float32)
        feed = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64), "feats": f}
        return np.mean([p.sess.run(["probs"], feed)[0] for p in self.ps], 0)


def latency_ens(model_dirs, out, n=100, threads=8):
    torch.set_num_threads(threads)
    states = [e.state for e in P.dev(T)]
    pr = EnsemblePredictor(model_dirs, threads)
    times = []
    for i in range(n + 3):
        t0 = time.perf_counter(); pr.probs([states[i % len(states)]]); times.append((time.perf_counter() - t0) * 1000)
    c = times[3:]
    res = {"task": T, "model_dirs": [str(d) for d in model_dirs], "device": "cpu", "threads": threads, "runtime": "onnxruntime",
           "notes": "experiments/summeval/export_mh.py latency_ens: protocol of benchmarks.v2.latency (dev texts, batch 1, n=100 after 3 warm-up); one request = shared rendering/tokenisation/features + every ensemble member's ONNX session in sequence, all 4 heads",
           "cpu_p50_ms": float(np.percentile(c, 50)), "cpu_p95_ms": float(np.percentile(c, 95)), "n_requests": len(c), "load_1min": os.getloadavg()[0]}
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == "__main__" and sys.argv[1] == "latency_ens":
    latency_ens(sys.argv[3:], sys.argv[2], threads=8)
