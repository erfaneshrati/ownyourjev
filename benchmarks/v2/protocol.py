"""Benchmark v2 protocol: what an experiment may see.

Per task, a fixed human-label budget is drawn from the *training* split (never the test split):
  - seeds: 20 gold-labeled examples (e.g. few-shot examples in a labeling prompt)
  - dev:   200 gold-labeled examples (labeler/prompt choice, model selection, thresholds)
Everything else is unlabeled: ``pool()`` (training texts) and ``eval_inputs()`` (test texts) carry no gold.
Experiments may use the 220 gold examples however they like (prompting, selection, even training), but
selection estimates must not be computed on examples a model was trained on.

SMS Spam and SummEval have no training split: the 220 examples (SummEval: whole source documents) are
carved out of the evaluation set, removed from it, and Jev is re-scored on the remaining examples.
Those two tasks are transductive: the unlabeled evaluation texts are the training pool, so models must be
cross-fitted with ``eval_folds()`` (no model scores a text it was trained on).

Test gold labels are read only by ``benchmarks.v2.final_eval``, which logs every access.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from benchmarks.tasks import get_task
from ownyourjev.data import Example, kfold

N_SEEDS = 20
N_DEV = 200
RESULTS = Path(os.environ.get("OYJ_RESULTS_V2", Path(__file__).resolve().parents[1] / "results_v2"))
CROSS_FIT = {"sms_spam", "summeval"}


def _strip(exs: list[Example]) -> list[Example]:
    return [replace(e, gold=None, labels=None) for e in exs]


@lru_cache(maxsize=None)
def _splits(name: str) -> dict:
    task = get_task(name)
    rng = random.Random(f"ownyourjev-v2-{name}")
    if name in CROSS_FIT:
        ev = task.eval()
        if task.spec.grouped:
            groups = sorted({e.group for e in ev})
            rng.shuffle(groups)
            take, rows = set(), 0
            for g in groups:
                if rows >= N_SEEDS + N_DEV:
                    break
                take.add(g)
                rows += sum(e.group == g for e in ev)
            carved = [e for e in ev if e.group in take]
            rest = [e for e in ev if e.group not in take]
        else:
            idx = list(range(len(ev)))
            rng.shuffle(idx)
            cut = set(idx[: N_SEEDS + N_DEV])
            carved = [e for i, e in enumerate(ev) if i in cut]
            rest = [e for i, e in enumerate(ev) if i not in cut]
        rng.shuffle(carved)
        seeds, dev = carved[:N_SEEDS], carved[N_SEEDS:]
        return {"seeds": seeds, "dev": dev, "pool": rest, "eval": rest}
    pool = task.pool()
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    seeds = [pool[i] for i in idx[:N_SEEDS]]
    dev = [pool[i] for i in idx[N_SEEDS : N_SEEDS + N_DEV]]
    rest = [pool[i] for i in idx[N_SEEDS + N_DEV :]]
    return {"seeds": seeds, "dev": dev, "pool": rest, "eval": task.eval()}


def seeds(name: str) -> list[Example]:
    """20 gold examples; ``labels`` is set to the gold labels (ready to pass as few-shot seeds)."""
    return [replace(e, labels=dict(e.gold)) for e in _splits(name)["seeds"]]


def dev(name: str) -> list[Example]:
    """200 gold examples; ``labels`` is set to the gold labels."""
    return [replace(e, labels=dict(e.gold)) for e in _splits(name)["dev"]]


def pool(name: str) -> list[Example]:
    """Unlabeled training texts (for cross-fit tasks: the evaluation texts, without labels)."""
    return _strip(_splits(name)["pool"])


def eval_inputs(name: str) -> list[Example]:
    """Unlabeled evaluation texts, in the fixed order final_eval expects predictions in."""
    return _strip(_splits(name)["eval"])


def eval_folds(name: str, k: int = 5) -> list[tuple[list[int], list[int]]]:
    """Cross-fit folds over eval_inputs (grouped by source document for SummEval)."""
    task = get_task(name)
    return kfold(eval_inputs(name), k, seed=0, grouped=task.spec.grouped)


def task_spec(name: str):
    return get_task(name).spec


def write_split_manifest(name: str) -> Path:
    s = _splits(name)
    out = RESULTS / name / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: [e.uid for e in v] for k, v in s.items()}, indent=0))
    return out
