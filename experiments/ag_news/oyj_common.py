"""Shared helpers for the wins_choice re-certification (ag_news, financial_phrasebank, emotion).
Lives in experiments/ag_news/ and is imported by the other two task folders via sys.path."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.v2 import protocol  # noqa: E402
from ownyourjev.data import Example  # noqa: E402


def paths(task):
    D = ROOT / "data/v2" / task
    R = ROOT / "benchmarks/results_v2" / task
    D.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    return D, R


def save_jsonl(path, exs):
    with open(path, "w") as f:
        for e in exs:
            f.write(json.dumps({"uid": e.uid, "state": e.state, "labels": e.labels, "group": e.group, "meta": e.meta}) + "\n")


def load_jsonl(path):
    out = []
    for l in open(path):
        r = json.loads(l)
        out.append(Example(r["uid"], r["state"], r["labels"], None, r.get("group", ""), r.get("meta", {})))
    return out


def v1_silver(task, kind="silver"):
    """v1 LLM labels keyed by uid, gold dropped, seeds/dev uids removed."""
    held = {e.uid for e in protocol.seeds(task)} | {e.uid for e in protocol.dev(task)}
    out = {}
    for l in open(ROOT / "data/silver" / task / f"{kind}.jsonl"):
        r = json.loads(l)
        if r["uid"] in held or not r.get("labels"):
            continue
        out[r["uid"]] = Example(r["uid"], r["state"], r["labels"], None, r.get("group", ""), r.get("meta", {}))
    return out


def v1_on_dev(task, kind="silver"):
    """v1 labels for dev uids (for measuring that labeler's dev agreement only; never trained on)."""
    dv = {e.uid for e in protocol.dev(task)}
    out = {}
    for l in open(ROOT / "data/silver" / task / f"{kind}.jsonl"):
        r = json.loads(l)
        if r["uid"] in dv and r.get("labels"):
            out[r["uid"]] = r["labels"]["answer"]
    return out


def acc(labels_by_uid, dev):
    xs = [(labels_by_uid[e.uid], e.labels["answer"]) for e in dev if labels_by_uid.get(e.uid) is not None]
    return (float(np.mean([a == b for a, b in xs])) if xs else float("nan")), len(xs)


def run_label(task, examples, name, *, model=None, seeds=None, guidance=None, distribution=False, questions=None,
              temperature=0.0, extra_body=None, concurrency=32, split="dev", timeout=40.0):
    """Label examples with label_examples; saves data/v2/<task>/labels/<split>_<name>.jsonl. Returns list."""
    import asyncio
    from ownyourjev.label import label_examples
    from ownyourjev.llm import OpenRouterClient
    D, _ = paths(task)
    out = D / "labels" / f"{split}_{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    qs = questions or protocol.task_spec(task).parsed

    async def go():
        c = OpenRouterClient(model=model, temperature=temperature, extra_body=extra_body, concurrency=concurrency, timeout=timeout)
        try:
            return await label_examples(c, qs, examples, purpose=f"{task}:{split}_{name}", seeds=seeds, guidance=guidance,
                                        distribution=distribution, progress=False)
        finally:
            await c.aclose()
    res = asyncio.run(go())
    save_jsonl(out, res)
    return res


def report(task, res, dev=None, name=""):
    """Agreement with dev gold, overall and on halves A (even idx) / B (odd idx)."""
    dev = dev or protocol.dev(task)
    lab = {e.uid: (e.labels or {}).get("answer") for e in res}
    a, n = acc(lab, dev)
    ha, _ = acc(lab, dev[0::2]); hb, _ = acc(lab, dev[1::2])
    print(f"{task:22s} {name:28s} dev={a:.3f} (n={n}) A={ha:.3f} B={hb:.3f}", flush=True)
    return a, ha, hb
