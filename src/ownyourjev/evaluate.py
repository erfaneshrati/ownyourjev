"""Metrics for the three question types, mirroring jev-benchmarking's metrics so numbers are comparable.

All functions take gold labels in canonical form and model probabilities:
- choice: probs (N, K) over ``question.options``; gold = option key
- score:  probs (N, K) over levels; gold = level index (float allowed); answer = expected level
- noul:   probs (N,) = P(true); gold = bool
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ownyourjev.spec import Question

N_BINS = 10


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return math.nan
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def _safe(fn: Callable[[], float]) -> float:
    try:
        v = float(fn())
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def choice_metrics(golds: list[str], probs: np.ndarray, options: list[str]) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    idx = {o: i for i, o in enumerate(options)}
    y = np.array([idx[g] for g in golds])
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    top_p = probs.max(1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": _safe(lambda: f1_score(y, pred, average="macro", zero_division=0)),
        "ece": ece(top_p, correct),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "mean_top_prob": float(top_p.mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
    }


def binary_metrics(golds: list[bool], p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(golds, dtype=bool)
    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "auroc": _safe(lambda: roc_auc_score(y, p)),
        "auprc": _safe(lambda: average_precision_score(y, p)),
        "ece": ece(conf, correct),
        "brier": float(((p - y) ** 2).mean()),
        "nll": float(-(y * np.log(np.clip(p, 1e-12, 1)) + (~y) * np.log(np.clip(1 - p, 1e-12, 1))).mean()),
        "positive_rate": float(y.mean()),
    }


def score_metrics(
    golds: list[float], probs: np.ndarray, groups: list[str] | None = None
) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(golds, dtype=float)
    k = probs.shape[1]
    s = probs @ np.arange(k)
    argmax = probs.argmax(1)
    out = {
        "spearman": _safe(lambda: stats.spearmanr(y, s).statistic),
        "pearson": _safe(lambda: stats.pearsonr(y, s).statistic),
        "kendall": _safe(lambda: stats.kendalltau(y, s).statistic),
        "mae": float(np.abs(y - s).mean()),
        "argmax_accuracy": float((argmax == np.round(y)).mean()),
        "argmax_mae": float(np.abs(argmax - y).mean()),
    }
    if groups is not None:
        by = defaultdict(list)
        for g, yi, si in zip(groups, y, s):
            by[g].append((yi, si))
        rhos = []
        for pairs in by.values():
            if len(pairs) < 3:
                continue
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            if a.std() == 0 or b.std() == 0:
                continue
            rhos.append(stats.spearmanr(a, b).statistic)
        out["group_spearman"] = float(np.mean(rhos)) if rhos else math.nan
    return out


def evaluate(question: Question, golds: list[Any], probs: np.ndarray, groups: list[str] | None = None) -> dict:
    if question.type == "choice":
        return choice_metrics(golds, probs, question.options)
    if question.type == "noul":
        return binary_metrics(golds, probs)
    return score_metrics(golds, probs, groups)


def primary_metric_name(question: Question) -> str:
    return {"choice": "accuracy", "noul": "f1", "score": "spearman"}[question.type]


def bootstrap_ci(
    fn: Callable[[np.ndarray], float], n: int, *, iters: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """95% percentile bootstrap of a statistic computed from a vector of resampled indices."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def evaluate_with_ci(
    question: Question, golds: list[Any], probs: np.ndarray, groups: list[str] | None = None, *, iters: int = 1000, metric: str | None = None
) -> dict:
    """Full metric dict plus a 95% CI for ``metric`` (default: the question type's primary metric)."""
    probs = np.asarray(probs)
    m = evaluate(question, golds, probs, groups)
    name = metric or primary_metric_name(question)
    if name not in m and question.type == "score" and groups is not None:
        name = "group_spearman"
    if name not in m:
        name = primary_metric_name(question)
    golds_arr = np.asarray(golds, dtype=object)
    groups_arr = np.asarray(groups, dtype=object) if groups is not None else None

    def stat(idx: np.ndarray) -> float:
        g = list(golds_arr[idx])
        gr = list(groups_arr[idx]) if groups_arr is not None else None
        return evaluate(question, g, probs[idx], gr)[name]

    m["primary"] = name
    m["primary_value"] = m[name]
    m["ci95"] = bootstrap_ci(stat, len(golds), iters=iters)
    return m
