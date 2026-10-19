"""Where do re-predicted and frozen test probs differ? (no labels)"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
ev = protocol.eval_inputs("paws"); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
pt = np.load(ROOT / "experiments/paws/calibration/test_repred.npy")
diff = np.abs(pt - frozen); L = np.array([len(render_state(e.state)) for e in ev])
print("frozen dtype", frozen.dtype, frozen.shape, "corr", np.corrcoef(pt, frozen)[0, 1])
print("quantiles diff", np.quantile(diff, [.5, .9, .99]))
order = np.argsort(L); rank = np.empty_like(order); rank[order] = np.arange(len(L))
for lo, hi in [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]:
    m = (rank >= lo) & (rank < hi); print("length-rank", lo, hi, "mean diff", diff[m].mean())
for lo in range(0, 8000, 2000): print("index", lo, diff[lo:lo + 2000].mean())
print("frac logits shift sign: mean(frozen-pt)", float((frozen - pt).mean()))
