"""Pick one threshold on dev (int8 probs of all 5 fold models, mean F1) and assemble the eval probs."""
import json
from common import *
T = "sms_spam"; M = ROOT / "data/v2/sms_spam/models/gte_full_e3"
dev = protocol.dev(T); y = [e.labels["answer"] for e in dev]; ev = protocol.eval_inputs(T)
P = [np.load(M / f"fold{k}/dev_probs_int8.npy") for k in range(5)]
grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
tab = {float(t): float(np.mean([acc_f1(y, p, t)[1] for p in P])) for t in grid}
for t, f in tab.items(): print(t, round(f, 4))
best = max(tab.values()); cands = [t for t, f in tab.items() if f >= best - 1e-9]
thr = min(cands, key=lambda t: abs(t - 0.5))  # tie-break: closest to 0.5
print("chosen thr", thr, "mean dev F1", tab[thr])
out = np.full(len(ev), np.nan, dtype=np.float32)
for k in range(5):
    idx = np.load(M / f"fold{k}/te_idx.npy"); out[idx] = np.load(M / f"fold{k}/te_probs_int8.npy")
assert not np.isnan(out).any()
np.save(ROOT / "data/v2/sms_spam/eval_probs.npy", out)
json.dump({"thr_table": tab, "threshold": thr, "per_fold_dev_f1_int8": [acc_f1(y, p, thr)[1] for p in P], "per_fold_dev_acc_int8": [acc_f1(y, p, thr)[0] for p in P],
           "per_fold_dev_f1_torch": [acc_f1(y, np.load(M / f"fold{k}/dev_probs.npy"))[1] for k in range(5)], "eval_pos_rate": float((out >= thr).mean())},
          open(ROOT / "data/v2/sms_spam/dev_summary.json", "w"), indent=1)
print("eval predicted positive rate", (out >= thr).mean())
