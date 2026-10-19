"""Experiment ledger for the paper: one JSON line per experiment, successes and failures alike.

Usage (from any experiment script or a shell one-liner):

    from benchmarks.v2.ledger import log_experiment
    log_experiment(
        task="boolq", round=1, kind="labeling",            # labeling|data|training|quantisation|latency|selection|other
        idea="LLM-generated questions per passage (5 per passage, balanced yes/no)",
        change="added 23.7k synthetic QA pairs to 9.2k pool labels",
        dev={"accuracy": 0.775, "baseline_accuracy": 0.835, "n": 200},
        cost_usd=0.21, gpu_minutes=14,
        outcome="failure",                                  # success|failure|neutral
        takeaway="synthetic questions too easy / passage-driven vs real queries; hurt gte",
    )

Lines go to benchmarks/results_v2/<task>/experiments.jsonl (one file per task, append-only, file-locked).
Never put test numbers in the ledger except via final_eval (which logs test scorings itself).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
from typing import Any

from benchmarks.v2.protocol import RESULTS

KINDS = {"labeling", "data", "training", "quantisation", "latency", "selection", "other"}
OUTCOMES = {"success", "failure", "neutral"}


def log_experiment(*, task: str, idea: str, outcome: str, kind: str = "other", round: int = 1, change: str = "",
                   dev: dict[str, Any] | None = None, cost_usd: float = 0.0, gpu_minutes: float = 0.0,
                   takeaway: str = "", **extra: Any) -> dict:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "task": task, "round": round, "kind": kind, "idea": idea,
           "change": change, "dev": dev or {}, "cost_usd": cost_usd, "gpu_minutes": gpu_minutes, "outcome": outcome,
           "takeaway": takeaway, **extra}
    p = RESULTS / task / "experiments.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, default=float) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return rec


def load(task: str | None = None) -> list[dict]:
    files = [RESULTS / task / "experiments.jsonl"] if task else sorted(RESULTS.glob("*/experiments.jsonl"))
    out = []
    for p in files:
        if p.is_file():
            out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


if __name__ == "__main__":  # CLI: python -m benchmarks.v2.ledger '{"task": ..., "idea": ..., "outcome": ...}'
    import sys

    print(log_experiment(**json.loads(sys.argv[1])))
