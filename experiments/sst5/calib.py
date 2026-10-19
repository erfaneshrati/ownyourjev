import numpy as np, itertools
def fit_bias(p, y, iters=3, grid=np.linspace(-3, 3, 61)):
    """coordinate ascent on additive log-biases maximising accuracy (ties -> smaller |b|)"""
    lp = np.log(np.clip(p, 1e-4, 1)); b = np.zeros(p.shape[1])
    for _ in range(iters):
        for k in range(1, p.shape[1]):
            best = None
            for g in grid:
                bb = b.copy(); bb[k] = g
                acc = ((lp + bb).argmax(1) == y).mean()
                key = (acc, -abs(g))
                if best is None or key > best[0]: best = (key, g)
            b[k] = best[1]
    return b
def apply(p, b): return (np.log(np.clip(p, 1e-4, 1)) + b).argmax(1)
def cv_acc(p, y, k=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); folds = np.array_split(idx, k); correct = 0
    for f in folds:
        tr = np.setdiff1d(idx, f); b = fit_bias(p[tr], y[tr]); correct += (apply(p[f], b) == y[f]).sum()
    return correct / len(y)
def cv_rep(p, y, seeds=(0,1,2)): return float(np.mean([cv_acc(p, y, seed=s) for s in seeds]))
