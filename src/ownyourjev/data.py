"""Example containers, jsonl IO, subsampling ladders, splits and cross-fit folds."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class Example:
    uid: str
    state: Any
    labels: dict[str, Any] | None = None  # canonical labels per question id (None = unlabeled / invalid)
    gold: dict[str, Any] | None = None  # gold labels when known (benchmarks only; never used for silver training)
    group: str = ""  # for grouped metrics / grouped folds (e.g. source document)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.labels is not None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Example":
        return Example(**json.loads(s))


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def read_jsonl(path: str | Path) -> list[Example]:
    with Path(path).open(encoding="utf-8") as f:
        return [Example.from_json(line) for line in f if line.strip()]


def stratified_subsample(
    examples: list[Example], n: int, *, key: Callable[[Example], Any] | None = None, seed: int = 0
) -> list[Example]:
    """Deterministic subsample of size n, stratified by ``key`` (round-robin over strata)."""
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    if key is None:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        return [examples[i] for i in sorted(idx[:n])]
    buckets: dict[Any, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[key(ex)].append(ex)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[Example] = []
    keys = sorted(buckets, key=lambda k: str(k))
    rng.shuffle(keys)
    while len(out) < n:
        progressed = False
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop())
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
    return out


def train_val_split(
    examples: list[Example], val_frac: float = 0.1, *, seed: int = 0, min_val: int = 16
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(min_val, int(round(len(examples) * val_frac)))
    n_val = min(n_val, max(1, len(examples) // 2))
    val = {i for i in idx[:n_val]}
    return [ex for i, ex in enumerate(examples) if i not in val], [ex for i, ex in enumerate(examples) if i in val]


def kfold(examples: list[Example], k: int, *, seed: int = 0, grouped: bool = False) -> list[tuple[list[int], list[int]]]:
    """K folds as (train_indices, held_out_indices). With ``grouped`` the ``group`` field never straddles folds."""
    rng = random.Random(seed)
    if grouped:
        groups = sorted({ex.group for ex in examples})
        rng.shuffle(groups)
        fold_of_group = {g: i % k for i, g in enumerate(groups)}
        fold_of = [fold_of_group[ex.group] for ex in examples]
    else:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        fold_of = [0] * len(examples)
        for pos, i in enumerate(idx):
            fold_of[i] = pos % k
    folds = []
    for f in range(k):
        held = [i for i, ff in enumerate(fold_of) if ff == f]
        train = [i for i, ff in enumerate(fold_of) if ff != f]
        folds.append((train, held))
    return folds


def label_key(qid: str) -> Callable[[Example], Any]:
    return lambda ex: (ex.labels or {}).get(qid)
