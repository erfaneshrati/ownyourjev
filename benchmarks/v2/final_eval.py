"""The only code that reads test gold labels. Every call is appended to results_v2/TEST_ACCESS.log.

Usage: uv run python -m benchmarks.v2.final_eval <task> [--rerun-reason "..."]

Reads results_v2/<task>/FROZEN.json, written by the experiment *before* it ever sees a test score:
{
  "description": "what the frozen recipe is and why it was chosen (dev evidence)",
  "heads": {"<qid>": {"probs": "path/to/eval_probs.npy", "threshold": 0.5}},   # one per question id
  "dev_metrics": {...}, "latency": "path/to/latency.json", "label_spend_usd": 0.0
}
probs are aligned with protocol.eval_inputs(task): (N,) P(true) for noul, (N, K) over question.options otherwise.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import typer

from benchmarks.run import jev_answers, jev_probs_matrix
from benchmarks.tasks import get_task
from benchmarks.v2 import protocol
from ownyourjev.evaluate import bootstrap_ci, evaluate

app = typer.Typer()
LOG = protocol.RESULTS / "TEST_ACCESS.log"
JEV_P95_FLOOR_MS = 70.0  # Jev's best published end-to-end latency (70-500 ms); our gate: CPU p95, 4 threads

# what Jev's published headline number is, per task: (head or "*mean*", metric)
HEADLINE = {
    "summeval": ("*mean*", "group_spearman"),
    "helpsteer2": ("*mean*", "spearman"),
}


def _metric_for(task_name: str, head: str, q) -> str:
    spec = get_task(task_name).spec
    if head == spec.primary_head:
        return {"group_spearman": "group_spearman", "mean_spearman": "spearman"}.get(spec.primary_metric, spec.primary_metric)
    return {"noul": "f1", "choice": "accuracy", "score": "spearman"}[q.type]


def _gold_eval(task_name: str):
    """Gold labels of the evaluation split, in eval_inputs order."""
    s = protocol._splits(task_name)
    return s["eval"]


@app.command()
def main(task_name: str, rerun_reason: str = typer.Option("", help="required if a final result already exists")):
    out_dir = protocol.RESULTS / task_name
    frozen_p = out_dir / "FROZEN.json"
    final_p = out_dir / "final.json"
    if not frozen_p.is_file():
        raise SystemExit(f"{frozen_p} missing: freeze the recipe (chosen on dev) before scoring on test")
    if final_p.is_file() and not rerun_reason:
        raise SystemExit(f"{final_p} exists: test was already scored once; pass --rerun-reason to re-score (logged)")
    if final_p.is_file():  # keep every earlier round's result: nothing a later round does can erase a test look
        n = 1 + len(list(out_dir.glob("round*_final.json")))
        prev = json.loads(final_p.read_text())
        (out_dir / f"round{n}_final.json").write_text(json.dumps(prev, indent=1, default=float))
        (out_dir / f"round{n}_FROZEN.json").write_text(json.dumps(prev.get("frozen", {}), indent=1, default=float))
    frozen = json.loads(frozen_p.read_text())
    task = get_task(task_name)
    spec = task.spec
    questions = spec.parsed
    ev = _gold_eval(task_name)
    groups = [e.group for e in ev] if spec.grouped else None
    per_head, probs_by_head = {}, {}
    for h, cfg in frozen["heads"].items():
        q = questions[h]
        p = np.load(cfg["probs"])
        exp = (len(ev),) if q.type == "noul" else (len(ev), q.num_outputs)
        if p.shape != exp:
            raise SystemExit(f"head {h}: probs shape {p.shape} != {exp}")
        golds = [e.gold[h] for e in ev]
        thr = float(cfg.get("threshold", 0.5))
        if q.type == "noul":
            from ownyourjev.evaluate import binary_metrics

            m = binary_metrics(golds, p, threshold=thr)
        else:
            m = evaluate(q, golds, p, groups)
        name = _metric_for(task_name, h, q)
        g_arr = np.asarray(golds, dtype=object)
        gr_arr = np.asarray(groups, dtype=object) if groups else None

        def stat(idx, q=q, p=p, g_arr=g_arr, gr_arr=gr_arr, name=name, thr=thr):
            gg = list(g_arr[idx])
            if q.type == "noul":
                from ownyourjev.evaluate import binary_metrics

                return binary_metrics(gg, p[idx], threshold=thr)[name]
            return evaluate(q, gg, p[idx], list(gr_arr[idx]) if gr_arr is not None else None)[name]

        m["metric"] = name
        m["ci95"] = bootstrap_ci(stat, len(ev), iters=1000)
        m["threshold"] = thr
        per_head[h] = m
        probs_by_head[h] = p
    # Jev on exactly the same examples, from the paper's per-example responses
    jev = {}
    for h, q in questions.items():
        jm = jev_probs_matrix(q, jev_answers(spec.questions, ev, h), ev)
        if jm is not None:
            jm_metrics = evaluate(q, [e.gold[h] for e in ev], jm, groups)
            jm_metrics["metric"] = _metric_for(task_name, h, q)
            jev[h] = jm_metrics
    head, metric = HEADLINE.get(task_name, (spec.primary_head, None))
    if head == "*mean*":
        ours = float(np.mean([per_head[h][metric] for h in questions if h in per_head]))
        theirs = float(np.mean([jev[h][metric] for h in questions if h in jev])) if jev else math.nan
        ours_ci = (math.nan, math.nan)
    else:
        metric = per_head[head]["metric"]
        ours, ours_ci = per_head[head][metric], per_head[head]["ci95"]
        theirs = jev.get(head, {}).get(metric, math.nan)
    lat = json.loads(Path(frozen["latency"]).read_text()) if frozen.get("latency") else {}
    result = {
        "task": task_name,
        "n_eval": len(ev),
        "headline": {"metric": metric, "head": head, "ours": ours, "ours_ci95": ours_ci, "jev_same_examples": theirs, "beats_jev": bool(ours > theirs)},
        "heads": per_head,
        "jev_same_examples": jev,
        "latency": lat,
        "latency_gate_ms": JEV_P95_FLOOR_MS,
        "latency_ok": bool(lat.get("cpu_p95_ms", math.inf) < JEV_P95_FLOOR_MS),
        "frozen": frozen,
        "scored_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rerun_reason": rerun_reason or None,
    }
    final_p.write_text(json.dumps(result, indent=1, default=float))
    digest = hashlib.sha256(frozen_p.read_bytes()).hexdigest()[:12]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{result['scored_at']}\t{task_name}\tfrozen={digest}\tours={ours:.4f}\tjev={theirs:.4f}\trerun_reason={rerun_reason!r}\n")
    print(json.dumps(result["headline"], default=float))
    print("latency_ok:", result["latency_ok"], lat)


if __name__ == "__main__":
    app()
