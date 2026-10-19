"""Post-processing used in the deployed pipeline (constant-time numpy on the (5 heads, 5 levels) output)."""
import numpy as np

def blend_hc(P):
    """helpfulness and correctness are two noisy estimates of nearly the same human judgement (gold r=0.95):
    give both heads the average of their expected scores (as a 2-point distribution with that mean)."""
    P = np.array(P, dtype=np.float64, copy=True)
    single = P.ndim == 2
    if single: P = P[None]
    lv = np.arange(5)
    s = (P[:, 0] @ lv + P[:, 1] @ lv) / 2
    lo = np.minimum(np.floor(s).astype(int), 3); f = s - lo
    Q = np.zeros((len(s), 5)); Q[np.arange(len(s)), lo] = 1 - f; Q[np.arange(len(s)), lo + 1] = f
    P[:, 0] = Q; P[:, 1] = Q
    return P[0] if single else P
