"""Shared helpers for the v2 noul re-certification experiments (copied per task folder)."""
import asyncio, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.v2 import protocol  # noqa: E402
from ownyourjev.data import Example  # noqa: E402
from ownyourjev.spec import render_state  # noqa: E402


def Q(task):
    return protocol.task_spec(task).parsed


def save_jsonl(path, exs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in exs:
            f.write(json.dumps({"uid": e.uid, "state": e.state, "labels": e.labels, "group": e.group, "meta": e.meta}) + "\n")


def load_jsonl(path):
    out = []
    for l in open(path):
        r = json.loads(l)
        out.append(Example(r["uid"], r["state"], r["labels"], None, r.get("group", ""), r.get("meta", {})))
    return out


def load_v1(task, variant):
    """v1 LLM labels keyed by uid; gold dropped; seeds/dev removed; (cross-fit) restricted to eval_inputs."""
    held = {e.uid for e in protocol.seeds(task)} | {e.uid for e in protocol.dev(task)}
    allowed = {e.uid for e in protocol.pool(task)}
    out = {}
    for l in open(ROOT / f"data/silver/{task}/{variant}.jsonl"):
        r = json.loads(l)
        if r["uid"] in held or r["uid"] not in allowed or not r.get("labels"):
            continue
        out[r["uid"]] = r["labels"]
    return out


def acc_f1(y, p, thr=0.5):
    y = np.asarray(y, bool); pr = np.asarray(p) >= thr
    tp = (y & pr).sum(); fp = (~y & pr).sum(); fn = (y & ~pr).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return float((y == pr).mean()), float(f1)


def best_thr(y, p, metric="f1", grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.025), 3)
    sc = [(acc_f1(y, p, t)[1 if metric == "f1" else 0], -abs(t - 0.5), t) for t in grid]
    return max(sc)[2]


def run(coro):
    return asyncio.run(coro)


def train_save(task, train_ex, val_ex, cfg, out_dir):
    from ownyourjev.train import train_with_oom_retry
    q = Q(task)["answer"]
    res = train_with_oom_retry(q, train_ex, val_ex, cfg)
    out_dir = Path(out_dir)
    res.model.save(out_dir, res.tokenizer)
    (out_dir / "train_summary.json").write_text(json.dumps({**res.summary(), "history": res.history, "config": cfg.to_dict()}, indent=1, default=float))
    return res


def torch_probs(res, exs):
    from ownyourjev.train import predict_probs
    return predict_probs(res.model, res.tokenizer, exs).reshape(-1)


def export_int8(model_dir):
    from ownyourjev.export import export_onnx, quantize_int8
    model_dir = Path(model_dir)
    p = export_onnx(model_dir)
    q = quantize_int8(p)
    return q


def onnx_probs(model_dir, exs, threads=8, batch=32, onnx_path=None):
    from ownyourjev.export import OnnxPredictor
    pr = OnnxPredictor(model_dir, onnx_path=onnx_path, threads=threads)
    states = [e.state for e in exs]
    order = sorted(range(len(states)), key=lambda i: len(render_state(states[i])))
    out = np.zeros(len(states), dtype=np.float32)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        out[idx] = pr.probs([states[i] for i in idx]).reshape(-1)
    return out
