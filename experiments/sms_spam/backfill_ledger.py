import sys; sys.path.insert(0, ".")
from benchmarks.v2.ledger import log_experiment as L
S = "sms_spam"
L(task=S, kind="labeling", idea="reuse v1 plain vs v1 seeded LLM labels (deepseek-v4.1-flash)", change="dev agreement of both v1 label sets",
  dev={"plain_f1": 0.909, "plain_acc": 0.980, "seeded_f1": 0.957, "seeded_acc": 0.990, "n": 200}, outcome="success",
  takeaway="seeded labels (v1 seeds) better on dev; reused for 5,334 eval texts at no cost")
L(task=S, kind="labeling", idea="label the 20 eval texts lacking v1 labels with v2 seeds", change="20 LLM calls", cost_usd=0.0007, outcome="neutral",
  takeaway="completes label coverage of the cross-fit pool")
L(task=S, kind="training", idea="gte-modernbert full FT lr 3e-5 vs 5e-5 (fold 0)", change="lr 5e-5", dev={"f1_lr3e-5": 0.979, "f1_lr5e-5": 0.957, "n": 200},
  gpu_minutes=6, outcome="failure", takeaway="higher lr not better (1 dev example); kept 3e-5")
L(task=S, kind="training", idea="5-fold cross-fit gte-modernbert full FT on v1 seeded labels", change="3 epochs, max_len 128",
  dev={"f1_per_fold_torch": [0.979, 0.979, 0.957, 0.979, 0.979], "baseline_labeler_f1": 0.957, "n": 200}, gpu_minutes=15, outcome="success",
  takeaway="student exceeds its labeler on dev")
L(task=S, kind="quantisation", idea="ONNX dynamic int8 per-channel", change="int8 export of all 5 folds",
  dev={"f1_mean_int8": 0.966, "f1_mean_torch": 0.975, "n": 200}, outcome="success", takeaway="~1 dev example lost on 2 folds; p95 31.7 ms CPU")
L(task=S, kind="selection", idea="threshold sweep on dev (mean F1 over 5 int8 fold models)", change="threshold kept 0.5",
  dev={"f1_at_0.5": 0.966, "f1_at_0.6": 0.970}, outcome="neutral", takeaway="flat 0.40-0.55; 0.60 gain = 1 example, treated as noise")
