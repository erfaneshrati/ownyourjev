"""Audited one-off: recompute the oracle rows of every <task>/calibration.json with the restricted oracle.

The first calibration scorings (before ~13:46 on 2026-10-19) used an oracle that could pick vector scaling or
one-vs-rest isotonic on multiclass heads, which changes the top choice (it re-classifies rather than recalibrates).
`calibration_eval.oracle` now uses the same argmax-preserving method set as the deployed calibrators. This script
re-applies it to every task so all oracle rows are like-for-like. Only the oracle rows change (they are upper bounds on
the test labels, never a reported model); the previous values are kept under `*_oracle_unrestricted`.
Reads test labels -> logged to TEST_ACCESS.log.

Usage: uv run python -m benchmarks.v2.oracle_recompute
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np

from benchmarks.run import jev_answers, jev_probs_matrix
from benchmarks.tasks import get_task
from benchmarks.v2 import protocol as P
from benchmarks.v2.calibration_eval import gold_index, oracle
from benchmarks.v2.calibration_report import TASKS
from ownyourjev.calibration import calibration_metrics


def main() -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    done = []
    for task in TASKS:
        d = P.RESULTS / task
        out_p = d / "calibration.json"
        if not out_p.is_file():
            continue
        res = json.loads(out_p.read_text())
        frozen = json.loads((d / "FROZEN.json").read_text())
        spec = get_task(task).spec
        ev = P._splits(task)["eval"]
        for h, row in res["heads"].items():
            q = spec.parsed[h]
            p = np.load(frozen["heads"][h]["probs"])
            y = np.array([gold_index(q, e.gold[h]) for e in ev])
            for key, probs in (("ours", p), ("jev", jev_probs_matrix(q, jev_answers(spec.questions, ev, h), ev))):
                k = f"{key}_oracle_recalibrated"
                if probs is None or k not in row:
                    continue
                row.setdefault(f"{key}_oracle_unrestricted", row[k])
                row[k] = calibration_metrics(oracle(probs, y), y)
            b1 = d / "BATCH1.json"
            if b1.is_file():  # same oracle on the batch-1 (served) predictions
                p1 = np.load(json.loads(b1.read_text())["heads"][h]["probs"])
                row["ours_b1_oracle_recalibrated"] = calibration_metrics(oracle(p1, y), y)
        res["oracle_recomputed_at"] = now
        out_p.write_text(json.dumps(res, indent=1, default=float))
        done.append(task)
    with (P.RESULTS / "TEST_ACCESS.log").open("a") as f:
        f.write(f"{now}\tALL\tanalysis=oracle_recompute\t(no model change; oracle rows re-fitted with argmax-preserving "
                f"methods on {','.join(done)})\n")
    print("recomputed:", done)


if __name__ == "__main__":
    main()
