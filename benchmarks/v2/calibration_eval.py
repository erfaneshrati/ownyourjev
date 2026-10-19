"""Audited calibration comparison with Jev (reads test labels; every call is logged to TEST_ACCESS.log).

Usage: uv run python -m benchmarks.v2.calibration_eval <task> [--rerun-reason "..."]

Requires benchmarks/results_v2/<task>/CALIBRATION.json, written BEFORE this script is run and fitted on dev
data only (never on test):
{
  "description": "how the calibrators were fitted (which dev predictions, out-of-sample how)",
  "heads": {"<qid>": {"method": "temperature|platt|vector|isotonic|none", "params": {...},
                      "dev_cv": {method: {nll, ece, brier, ...}}, "n_fit": 200}}
}
For every head of the task, on the frozen test predictions referenced by FROZEN.json, it reports:
  ours_raw, ours_calibrated (the dev-fitted calibrator), jev_raw (Jev as delivered), and, for both systems,
  *_oracle_recalibrated (recalibrated by 5-fold CV on the test labels: an upper bound on what any recalibration
  could achieve, applied identically to ours and to Jev).
Calibration never changes the deployed decisions (decisions use FROZEN.json thresholds on the raw probabilities),
so accuracy/F1 verdicts are unaffected.
Score heads are evaluated as a distribution over levels, with the gold level = the gold score rounded to the
nearest level.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import typer

from benchmarks.run import jev_answers, jev_probs_matrix
from benchmarks.tasks import get_task
from benchmarks.v2 import protocol as P
from ownyourjev.calibration import Calibrator, calibration_metrics, gold_index, select_calibrator

app = typer.Typer()


ALLOWED_MULTICLASS = ("none", "temperature")  # argmax-preserving: a served answer's choice must equal its argmax
ALLOWED_BINARY = ("none", "temperature", "platt", "isotonic")  # monotone maps (decision stays at the frozen threshold)


def oracle(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold recalibration fitted on the TEST labels (5-fold CV): an upper bound on what the allowed
    calibrators could do. Applied identically to ours and to Jev with the same allowed method set, so it never
    changes the top choice of a multiclass head (calibration, not re-classification); never a reported model."""
    from sklearn.model_selection import KFold

    methods = ALLOWED_BINARY if np.asarray(p).ndim == 1 else ALLOWED_MULTICLASS
    oof = np.zeros_like(p, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(p):
        cal, _ = select_calibrator(p[tr], y[tr], methods=methods, k=3)
        oof[te] = cal.apply(p[te])
    return oof


@app.command()
def main(task: str, rerun_reason: str = typer.Option("", help="required if calibration.json already exists")):
    d = P.RESULTS / task
    cal_p, out_p = d / "CALIBRATION.json", d / "calibration.json"
    if not cal_p.is_file():
        raise SystemExit(f"{cal_p} missing: fit and freeze calibrators on dev first")
    if out_p.is_file() and not rerun_reason:
        raise SystemExit(f"{out_p} exists: pass --rerun-reason to re-score (logged)")
    cal = json.loads(cal_p.read_text())
    frozen = json.loads((d / "FROZEN.json").read_text())
    spec = get_task(task).spec
    ev = P._splits(task)["eval"]
    res = {"task": task, "heads": {}, "description": cal.get("description", ""),
           "scored_at": dt.datetime.now().isoformat(timespec="seconds"), "rerun_reason": rerun_reason or None}
    for h, q in spec.parsed.items():
        if h not in frozen["heads"]:
            continue
        if h not in cal["heads"]:
            raise SystemExit(f"CALIBRATION.json has no entry for head {h!r}")
        p = np.load(frozen["heads"][h]["probs"])
        y = np.array([gold_index(q, e.gold[h]) for e in ev])
        c = Calibrator.from_dict(cal["heads"][h])
        allowed = ALLOWED_MULTICLASS if q.type != "noul" else ALLOWED_BINARY
        if c.method not in allowed:
            raise SystemExit(f"head {h!r}: method {c.method!r} not allowed for {q.type} heads (allowed: {allowed})")
        row = {"method": c.method, "ours_raw": calibration_metrics(p, y), "ours_calibrated": calibration_metrics(c.apply(p), y),
               "ours_oracle_recalibrated": calibration_metrics(oracle(p, y), y)}
        jm = jev_probs_matrix(q, jev_answers(spec.questions, ev, h), ev)
        if jm is not None:
            row["jev_raw"] = calibration_metrics(jm, y)
            row["jev_oracle_recalibrated"] = calibration_metrics(oracle(jm, y), y)
        res["heads"][h] = row
    out_p.write_text(json.dumps(res, indent=1, default=float))
    digest = hashlib.sha256(cal_p.read_bytes()).hexdigest()[:12]
    with (P.RESULTS / "TEST_ACCESS.log").open("a") as f:
        f.write(f"{res['scored_at']}\t{task}\tanalysis=calibration\tcalibration={digest}\trerun_reason={rerun_reason!r}\n")
    for h, r in res["heads"].items():
        line = f"{task}/{h} [{r['method']}] ECE ours raw {r['ours_raw']['ece']:.3f} -> cal {r['ours_calibrated']['ece']:.3f}"
        if "jev_raw" in r:
            line += f" | Jev raw {r['jev_raw']['ece']:.3f}, Jev oracle {r['jev_oracle_recalibrated']['ece']:.3f}"
        print(line)


if __name__ == "__main__":
    app()
