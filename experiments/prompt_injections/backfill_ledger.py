import sys; sys.path.insert(0, ".")
from benchmarks.v2.ledger import log_experiment as L
T = "prompt_injections"
L(task=T, kind="labeling", idea="labeler/prompt choice on dev (flash plain / +v2 seeds / +seeds+guidelines / soft / pro)", change="200 dev labels each",
  dev={"flash_plain": 0.895, "flash_seeded": 0.945, "flash_seeded_guide": 0.965, "flash_seeded_guide_soft": 0.960, "pro_seeded_guide": 0.935, "n": 200},
  cost_usd=0.05, outcome="success", takeaway="seeds +5 pts, seed-derived guidelines +2 pts; pro model worse")
L(task=T, kind="labeling", idea="guidelines v2 from dev half-A errors (role-play requests, ATTENTION-STOP, fake template fields)",
  change="3 extra rules", dev={"A": 1.0, "B": 0.96, "baseline_B": 0.97}, cost_usd=0.008, outcome="failure",
  takeaway="fits half A but not better on held-out half B; not adopted")
L(task=T, kind="data", idea="label 326 pool texts with flash+seeds+guide", change="pool labels", cost_usd=0.009, outcome="neutral", takeaway="110 positives")
L(task=T, kind="data", idea="seed-steered synthetic EN/DE inputs (6 sampling seeds x 200), relabeled by the same labeler, keep agreeing",
  change="+1,148 synthetic", cost_usd=0.14, dev={"gte_pool_only": 0.935, "gte_pool_synth_s0": 0.945, "gte_pool_synth_s1": 0.930},
  outcome="neutral", takeaway="small/noisy gain; synthetic alone 0.840 (domain gap)")
L(task=T, kind="training", idea="UltraBERT vs gte on pool+synth", change="backbone", dev={"ultrabert": 0.935, "gte": 0.945}, outcome="failure", takeaway="no gain, and int8 risk")
L(task=T, kind="training", idea="add the 20 seeds, 8 epochs", change="pool+synth+seeds, gte 8 ep", dev={"accuracy": 0.950, "e5": 0.940}, outcome="neutral",
  takeaway="best single-model dev; best threshold very low (0.075) -> labeler under-calls injections")
L(task=T, kind="selection", idea="5-fold CV on dev, no dev in training (baseline for dev-in-training)", change="", dev={"oof_acc@0.5": 0.935, "oof_acc@best_thr0.225": 0.945},
  gpu_minutes=10, outcome="neutral", takeaway="paired baseline for adding dev gold to training")
