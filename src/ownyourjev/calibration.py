"""Post-hoc probability calibration and calibration metrics.

Works on probabilities (what a deployed model or Jev returns), so it can calibrate any frozen model without
re-running it. Two shapes:
- binary (Noul): p of shape (N,) = P(true), labels y in {0, 1}
- multiclass (Choice / Score levels): P of shape (N, K), labels y in {0..K-1}

Calibrators (fit on held-out data only, e.g. the 200 gold dev examples via cross-validation):
- "none"
- "temperature": divide logits by T (keeps the argmax / the 0.5 decision of binary heads)
- "platt" (binary): sigmoid(a * logit + b);  "vector" (multiclass): softmax(logits / T + b_k)
- "isotonic": isotonic regression on the (top-class) probability; multiclass via one-vs-rest + renormalisation

`select_calibrator` picks the method with the lowest K-fold cross-validated NLL on the fitting data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------------------------------------- utils
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _log_probs(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, EPS, 1.0))


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def is_binary(p: np.ndarray) -> bool:
    return np.asarray(p).ndim == 1


# --------------------------------------------------------------------------------------------- metrics
def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    if is_binary(p):
        q = np.clip(p, EPS, 1 - EPS)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    if is_binary(p):
        return float(((p - y) ** 2).mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    return float(((p - onehot) ** 2).sum(1).mean())


def _conf_correct(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-label confidence and correctness (binary: decision at 0.5)."""
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    if is_binary(p):
        pred = (p >= 0.5).astype(int)
        return np.maximum(p, 1 - p), (pred == y).astype(float)
    return p.max(1), (p.argmax(1) == y).astype(float)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10, adaptive: bool = False) -> float:
    conf, corr = _conf_correct(p, y)
    if adaptive:  # equal-mass bins
        order = np.argsort(conf)
        bins = np.array_split(order, n_bins)
    else:
        idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
    tot = 0.0
    for b in bins:
        if len(b):
            tot += len(b) / len(conf) * abs(conf[b].mean() - corr[b].mean())
    return float(tot)


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf, corr = _conf_correct(p, y)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "n": int(m.sum()), "conf": float(conf[m].mean()), "acc": float(corr[m].mean())})
    return out


def calibration_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return {
        "ece": ece(p, y, 10),
        "ece15": ece(p, y, 15),
        "ace": ece(p, y, 10, adaptive=True),
        "brier": brier(p, y),
        "nll": nll(p, y),
        "mean_conf": float(_conf_correct(p, y)[0].mean()),
        "accuracy": float(_conf_correct(p, y)[1].mean()),
        "reliability": reliability(p, y, 10),
    }


# ------------------------------------------------------------------------------------------ calibrators
@dataclass
class Calibrator:
    method: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    # -- fitting
    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        p = np.asarray(p, float)
        y = np.asarray(y).astype(int)
        m = self.method
        if m == "none":
            self.params = {}
        elif m == "temperature":
            self.params = {"T": _fit_temperature(p, y)}
        elif m == "platt":
            assert is_binary(p), "platt is for binary heads; use 'vector' for multiclass"
            a, b = _fit_affine_binary(_logit(p), y)
            self.params = {"a": a, "b": b}
        elif m == "vector":
            assert not is_binary(p), "vector scaling is for multiclass heads"
            T, bias = _fit_vector(_log_probs(p), y)
            self.params = {"T": T, "bias": bias.tolist()}
        elif m == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            if is_binary(p):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
                self.params = {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}
            else:  # one-vs-rest per class, renormalised
                xs, ys = [], []
                for k in range(p.shape[1]):
                    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[:, k], (y == k).astype(float))
                    xs.append(ir.X_thresholds_.tolist())
                    ys.append(ir.y_thresholds_.tolist())
                self.params = {"x": xs, "y": ys}
        else:
            raise ValueError(f"unknown method {m}")
        return self

    # -- applying
    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        m, q = self.method, self.params
        if m == "none":
            return p
        if m == "temperature":
            if is_binary(p):
                return _sigmoid(_logit(p) / q["T"])
            return _softmax(_log_probs(p) / q["T"])
        if m == "platt":
            return _sigmoid(q["a"] * _logit(p) + q["b"])
        if m == "vector":
            return _softmax(_log_probs(p) / q["T"] + np.asarray(q["bias"])[None, :])
        if m == "isotonic":
            if is_binary(p):
                return np.clip(np.interp(p, q["x"], q["y"]), EPS, 1 - EPS)
            cols = [np.interp(p[:, k], q["x"][k], q["y"][k]) for k in range(p.shape[1])]
            P = np.clip(np.stack(cols, 1), EPS, None)
            return P / P.sum(1, keepdims=True)
        raise ValueError(m)

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> Calibrator:
        return Calibrator(d["method"], d.get("params", {}))


def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def loss(logT):
        return nll(Calibrator("temperature", {"T": math.exp(logT)}).apply(p), y)

    r = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(r.x))


def _fit_affine_binary(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize

    def loss(w):
        q = np.clip(_sigmoid(w[0] * z + w[1]), EPS, 1 - EPS)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean() + 1e-4 * (w[0] - 1) ** 2

    r = minimize(loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def _fit_vector(L: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    from scipy.optimize import minimize

    K = L.shape[1]

    def loss(w):
        T = math.exp(w[0])
        P = _softmax(L / T + w[1:][None, :])
        return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean() + 1e-3 * (w[1:] ** 2).sum()

    r = minimize(loss, x0=np.zeros(K + 1), method="L-BFGS-B")
    return float(math.exp(r.x[0])), r.x[1:]


# -------------------------------------------------------------------------------------------- selection
DEFAULT_METHODS_BINARY = ("none", "temperature", "platt", "isotonic")
DEFAULT_METHODS_MULTI = ("none", "temperature", "vector", "isotonic")


def cv_scores(p: np.ndarray, y: np.ndarray, methods=None, k: int = 5, seed: int = 0, groups=None) -> dict[str, dict]:
    """K-fold CV of each calibration method on the fitting data. Returns out-of-fold metrics per method."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    methods = methods or (DEFAULT_METHODS_BINARY if is_binary(p) else DEFAULT_METHODS_MULTI)
    if groups is not None:
        splits = list(GroupKFold(n_splits=k).split(p, y, groups))
    else:
        counts = np.bincount(y)
        if (counts[counts > 0] >= k).all() and len(counts[counts > 0]) > 1:
            splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(p if is_binary(p) else p[:, 0], y))
        else:
            splits = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(p))
    out = {}
    for m in methods:
        oof = np.zeros_like(p)
        ok = True
        for tr, te in splits:
            try:
                c = Calibrator(m).fit(p[tr], y[tr])
                oof[te] = c.apply(p[te])
            except (ValueError, AssertionError, np.linalg.LinAlgError):
                ok = False
                break
        if not ok:
            continue
        out[m] = {k2: v for k2, v in calibration_metrics(oof, y).items() if k2 != "reliability"}
    return out


def select_calibrator(p: np.ndarray, y: np.ndarray, methods=None, k: int = 5, seed: int = 0, groups=None,
                      criterion: str = "nll") -> tuple[Calibrator, dict]:
    """Pick the method with the best cross-validated `criterion`, then refit it on all the data."""
    scores = cv_scores(p, y, methods, k, seed, groups)
    best = min(scores, key=lambda m: scores[m][criterion])
    return Calibrator(best).fit(p, y), scores


# ---------------------------------------------------------------------------------- per-question helpers
ARGMAX_PRESERVING = ("none", "temperature")  # Choice / Score: a served answer's choice must equal its argmax


def gold_index(question, gold) -> int:
    """Label index of a gold answer: Noul -> 0/1, Choice -> option index, Score -> nearest level."""
    if question.type == "noul":
        return int(bool(gold))
    if question.type == "choice":
        return question.options.index(gold)
    return int(min(max(round(float(gold)), 0), question.num_outputs - 1))


def fit_for_question(question, p: np.ndarray, golds: list, k: int = 5) -> tuple[Calibrator, dict]:
    """Select and fit a calibrator for one question on held-out (dev) predictions.

    Noul heads may use any monotone map (decisions stay at the threshold on the raw probability); Choice and
    Score heads only argmax-preserving ones, so calibration never changes an answer."""
    y = np.array([gold_index(question, g) for g in golds])
    methods = DEFAULT_METHODS_BINARY if question.type == "noul" else ARGMAX_PRESERVING
    return select_calibrator(np.asarray(p, float), y, methods=methods, k=k)
