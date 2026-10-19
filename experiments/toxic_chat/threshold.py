"""Smoothed dev threshold: the t maximising mean F1 over bootstrap resamples of dev (less noisy than raw argmax)."""
import numpy as np
from sklearn.metrics import f1_score


def boot_threshold(y, p, iters=500, seed=0, grid=None):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(iters)]
    P = p[None, :] >= grid[:, None]  # (G, N)
    scores = np.zeros(len(grid))
    for ii in idx:
        yy = y[ii]; pp = P[:, ii]
        tp = (pp & yy).sum(1); fp = (pp & ~yy).sum(1); fn = (~pp & yy).sum(1)
        scores += np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    scores /= iters
    b = int(np.argmax(scores))
    return float(grid[b]), float(scores[b]), dict(zip(grid.tolist(), scores.round(4).tolist()))


def cv_boot_threshold_f1(y, p, k=5, reps=10, seed=0):
    """Honest F1 of the boot_threshold rule (threshold chosen on k-1 folds)."""
    y = np.asarray(y, bool); p = np.asarray(p, float)
    rng = np.random.default_rng(seed); res = []
    for r in range(reps):
        pos, neg = np.where(y)[0], np.where(~y)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        fold = np.empty(len(y), int); fold[pos] = np.arange(len(pos)) % k; fold[neg] = np.arange(len(neg)) % k
        pred = np.zeros(len(y), bool)
        for f in range(k):
            t, _, _ = boot_threshold(y[fold != f], p[fold != f], iters=100, seed=r)
            pred[fold == f] = p[fold == f] >= t
        res.append(f1_score(y, pred, zero_division=0))
    return float(np.mean(res))


def smooth_threshold(y, p, half_window=0.05, **kw):
    """boot_threshold curve, additionally averaged over a +-half_window threshold window, argmax."""
    _, _, curve = boot_threshold(y, p, **kw)
    ts = np.array(list(curve)); f = np.array(list(curve.values()))
    sm = np.array([f[np.abs(ts - t) <= half_window + 1e-9].mean() for t in ts])
    b = int(np.argmax(sm))
    return float(ts[b]), float(sm[b])
