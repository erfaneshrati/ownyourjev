"""Post-hoc stacking of the 5 head scores on gold dev: target attribute's score = ridge on the 5 expected scores.
Fitted on dev predictions of a model that never trained on dev; estimated by repeated K-fold CV on dev."""
import json, sys
import numpy as np
from sklearn.linear_model import Ridge
from scipy import stats

def feats(P):  # P: (N, 5 heads, 5 levels) -> (N, 5) expected scores
    return P @ np.arange(5)

def fit(X, y, alpha):
    return Ridge(alpha=alpha).fit(X, y)

def cv(P, G, alpha=10.0, k=5, reps=20, which=None):
    X = feats(P); N = len(X); out = np.zeros((reps, G.shape[1])); base = []
    for j in range(G.shape[1]):
        base.append(stats.spearmanr(X[:, j], G[:, j]).statistic)
    for r in range(reps):
        rng = np.random.default_rng(r); perm = rng.permutation(N); folds = np.array_split(perm, k)
        for j in range(G.shape[1]):
            pred = np.zeros(N)
            for f in folds:
                tr = np.setdiff1d(perm, f)
                m = fit(X[tr], G[tr, j], alpha); pred[f] = m.predict(X[f])
            out[r, j] = stats.spearmanr(pred, G[:, j]).statistic
    return np.array(base), out.mean(0)

def to_probs(s):  # expected score s -> 2-point distribution over levels with mean s
    s = np.clip(s, 0, 4); lo = np.floor(s).astype(int); lo = np.minimum(lo, 3); f = s - lo
    P = np.zeros((len(s), 5)); P[np.arange(len(s)), lo] = 1 - f; P[np.arange(len(s)), lo + 1] = f
    return P

def make(P, G, alpha, use):
    """use: list of bools per attribute (stack or keep raw). Returns json-able params."""
    X = feats(P); st = {"alpha": alpha, "coef": [], "intercept": [], "use": use}
    for j in range(G.shape[1]):
        m = fit(X, G[:, j], alpha); st["coef"].append(m.coef_.tolist()); st["intercept"].append(float(m.intercept_))
    return st

def apply(P, st):  # (N,5,5) -> (N,5,5)
    X = feats(P); out = P.copy()
    for j, u in enumerate(st["use"]):
        if u:
            out[:, j] = to_probs(X @ np.asarray(st["coef"][j]) + st["intercept"][j])
    return out

def apply_one(P, st):
    return apply(P[None], st)[0]
