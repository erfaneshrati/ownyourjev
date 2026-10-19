"""Paired comparison of two models on the same 200 dev examples (fp32 ONNX dev probs).
Metrics: Spearman(E[toxicity]), AUROC(E[toxicity]) and toxic accuracy with the threshold chosen by 2-fold CV
(threshold = argmax of 7-pt smoothed accuracy curve on one half, applied to the other), repeated over 50 random
half-splits. Paired bootstrap (2000) over examples for AUROC/Spearman; for CV-accuracy, paired over splits.
Usage: compare_dev.py NEW_MODEL OLD_MODEL"""
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "experiments/toxigen")
from common import DATA, TASK
from benchmarks.v2 import protocol as P

dev = P.dev(TASK); y = np.array([e.gold["toxic"] for e in dev]); g = np.array([float(e.gold["toxicity"]) for e in dev])
TS = np.arange(0.05, 0.8, 0.01)


def smooth_thr(s, yy):
    acc = np.array([((s >= t) == yy).mean() for t in TS]); sm = np.convolve(acc, np.ones(7) / 7, "same")
    sm[:3] = sm[-3:] = 0
    return TS[int(np.argmax(sm))]


def cv_acc(s, rng, n=50):
    out = []
    for _ in range(n):
        idx = rng.permutation(len(s)); A, B = idx[:100], idx[100:]
        tA, tB = smooth_thr(s[A], y[A]), smooth_thr(s[B], y[B])
        out.append((((s[B] >= tA) == y[B]).sum() + ((s[A] >= tB) == y[A]).sum()) / len(s))
    return np.array(out)


def es(name):
    p = np.load(DATA / "models" / name / "dev_probs_fp32onnx.npy"); return p[:, 1:] @ np.arange(5) / 4


a, b = es(sys.argv[1]), es(sys.argv[2])
rng = np.random.default_rng(0)
ca, cb = cv_acc(a, np.random.default_rng(1)), cv_acc(b, np.random.default_rng(1))
print(f"cv-acc  new {ca.mean():.4f}  old {cb.mean():.4f}  diff {np.mean(ca - cb):+.4f}  (splits new>old {np.mean(ca > cb):.2f})")
print(f"full-dev smoothed thr: new {smooth_thr(a, y):.2f} acc {((a >= smooth_thr(a, y)) == y).mean():.3f} | old {smooth_thr(b, y):.2f} acc {((b >= smooth_thr(b, y)) == y).mean():.3f}")
d_auc, d_sp = [], []
for _ in range(2000):
    i = rng.integers(0, 200, 200)
    if y[i].all() or not y[i].any():
        continue
    d_auc.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d_sp.append(stats.spearmanr(g[i], a[i]).statistic - stats.spearmanr(g[i], b[i]).statistic)
for nm, d, fa, fb in [("auroc", d_auc, roc_auc_score(y, a), roc_auc_score(y, b)), ("spearman", d_sp, stats.spearmanr(g, a).statistic, stats.spearmanr(g, b).statistic)]:
    d = np.array(d); print(f"{nm:8s} new {fa:.4f} old {fb:.4f} diff {fa - fb:+.4f}  95%CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  P(diff<=0) {np.mean(d <= 0):.3f}")
