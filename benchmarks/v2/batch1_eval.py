"""Audited serving-consistency check (reads test labels; every call is logged to TEST_ACCESS.log).

Background: our int8 ONNX models use dynamic activation quantization, whose scales are computed over the whole
input tensor -- so a text's probabilities depend on which other texts share its batch (and their padding). The
frozen test predictions of several tasks were computed in batches, while the model is served (and its latency
measured) one request at a time. This script re-scores the task with test predictions recomputed at batch size 1
-- the deployed serving mode -- with no other change (same artifact, preprocessing, decision rule).

Usage: uv run python -m benchmarks.v2.batch1_eval <task> [--rerun-reason "..."]
Requires benchmarks/results_v2/<task>/BATCH1.json, written before running:
{"description": "...", "heads": {"<qid>": {"probs": "path/to/eval_probs_b1.npy", "dev_probs": "path/to/dev_probs_b1.npy"}},
 "calibration": {"<qid>": {"method": ..., "params": ...}}}   # optional: calibrator refit on the batch-1 DEV predictions
Thresholds come from FROZEN.json (unchanged).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import typer

from benchmarks.tasks import get_task
from benchmarks.v2 import protocol as P
from benchmarks.v2.calibration_eval import ALLOWED_BINARY, ALLOWED_MULTICLASS, gold_index
from benchmarks.v2.final_eval import HEADLINE, _metric_for
from ownyourjev.calibration import Calibrator, calibration_metrics
from ownyourjev.evaluate import binary_metrics, bootstrap_ci, evaluate

app = typer.Typer()


@app.command()
def main(task: str, rerun_reason: str = typer.Option("")):
    d = P.RESULTS / task
    spec_p, out_p = d / "BATCH1.json", d / "batch1.json"
    if not spec_p.is_file():
        raise SystemExit(f"{spec_p} missing")
    if out_p.is_file() and not rerun_reason:
        raise SystemExit(f"{out_p} exists: pass --rerun-reason (logged)")
    b1 = json.loads(spec_p.read_text())
    frozen = json.loads((d / "FROZEN.json").read_text())
    final = json.loads((d / "final.json").read_text())
    spec = get_task(task).spec
    ev = P._splits(task)["eval"]
    groups = [e.group for e in ev] if spec.grouped else None
    res = {"task": task, "heads": {}, "description": b1.get("description", ""),
           "scored_at": dt.datetime.now().isoformat(timespec="seconds"), "rerun_reason": rerun_reason or None}
    for h, q in spec.parsed.items():
        if h not in frozen["heads"]:
            continue
        p0 = np.load(frozen["heads"][h]["probs"])
        p1 = np.load(b1["heads"][h]["probs"])
        if p1.shape != p0.shape:
            raise SystemExit(f"{h}: batch-1 probs shape {p1.shape} != frozen {p0.shape}")
        thr = float(frozen["heads"][h].get("threshold", 0.5) or 0.5)
        golds = [e.gold[h] for e in ev]
        name = _metric_for(task, h, q)
        if q.type == "noul":
            m = binary_metrics(golds, p1, threshold=thr)
            flips = float(((p1 >= thr) != (p0 >= thr)).mean())
        else:
            m = evaluate(q, golds, p1, groups)
            flips = float((p1.argmax(1) != p0.argmax(1)).mean())
        g_arr = np.asarray(golds, dtype=object)
        gr_arr = np.asarray(groups, dtype=object) if groups else None

        def stat(idx, q=q, p=p1, g_arr=g_arr, gr_arr=gr_arr, name=name, thr=thr):
            gg = list(g_arr[idx])
            if q.type == "noul":
                return binary_metrics(gg, p[idx], threshold=thr)[name]
            return evaluate(q, gg, p[idx], list(gr_arr[idx]) if gr_arr is not None else None)[name]

        y = np.array([gold_index(q, g) for g in golds])
        row = {"metric": name, "batch1": m[name], "batch1_ci95": bootstrap_ci(stat, len(ev), iters=1000),
               "frozen_batched": final["heads"][h][name], "decision_flips": flips,
               "max_abs_prob_diff": float(np.abs(p1 - p0).max()), "calib_raw": calibration_metrics(p1, y)}
        cal = (b1.get("calibration") or {}).get(h)
        if cal:
            allowed = ALLOWED_BINARY if q.type == "noul" else ALLOWED_MULTICLASS
            if cal["method"] not in allowed:
                raise SystemExit(f"{h}: calibration method {cal['method']} not allowed")
            row["calibration_method"] = cal["method"]
            row["calib_calibrated"] = calibration_metrics(Calibrator.from_dict(cal).apply(p1), y)
        res["heads"][h] = row
    head, _ = HEADLINE.get(task, (spec.primary_head, None))
    hrow = res["heads"][head]
    jev = final["headline"]["jev_same_examples"]
    lo, hi = hrow["batch1_ci95"]
    res["headline"] = {"head": head, "metric": hrow["metric"], "batch1": hrow["batch1"], "batch1_ci95": [lo, hi],
                       "frozen_batched": final["headline"]["ours"], "jev": jev,
                       "verdict_batch1": "win" if lo > jev else ("loss" if hi < jev else "tie")}
    out_p.write_text(json.dumps(res, indent=1, default=float))
    with (P.RESULTS / "TEST_ACCESS.log").open("a") as f:
        f.write(f"{res['scored_at']}\t{task}\tanalysis=batch1_serving\tours_b1={hrow['batch1']:.4f}\tjev={jev:.4f}\trerun_reason={rerun_reason!r}\n")
    print(json.dumps(res["headline"], default=float))
    for h, r in res["heads"].items():
        print(f"{h}: batch1 {r['batch1']:.4f} vs batched {r['frozen_batched']:.4f}; flips {r['decision_flips']:.3%}; max |dp| {r['max_abs_prob_diff']:.3f}")


if __name__ == "__main__":
    app()
