import sys; sys.path.insert(0, ".")
from benchmarks.v2.ledger import log_experiment as L
P = "paws"
L(task=P, kind="labeling", idea="labeler/prompt choice on dev: deepseek-v4.1-flash plain / +v2 seeds / +seeds+guidelines; deepseek-v4-pro +seeds",
  change="200 dev labels each", dev={"flash_plain": 0.830, "flash_seeded": 0.875, "flash_seeded_guide": 0.850, "pro_seeded": 0.835, "n": 200},
  cost_usd=0.04, outcome="success", takeaway="seeds help (+4.5 pts); hand-written swap guidelines hurt; the pro model is worse than flash")
L(task=P, kind="training", idea="gte-modernbert full FT on 4.5k v1 seeded labels", change="lr 3e-5, 3 ep",
  dev={"accuracy": 0.740, "baseline_labeler": 0.875}, gpu_minutes=2, outcome="failure", takeaway="gte cannot learn PAWS word-order from 4.5k labels")
L(task=P, kind="data", idea="train only on v1 plain/seeded agreement subset", change="4.0k labels", dev={"accuracy": 0.705, "baseline_accuracy": 0.740},
  gpu_minutes=1, outcome="failure", takeaway="filtering removes the hard (disagreeing) pairs that matter")
L(task=P, kind="training", idea="gte longer training lr 2e-5 8 ep", change="epochs/lr", dev={"accuracy": 0.735, "baseline_accuracy": 0.740},
  gpu_minutes=5, outcome="failure", takeaway="overfits after epoch 2")
L(task=P, kind="training", idea="UltraBERT full FT on 4.5k v1 seeded labels (teacher)", change="backbone ultrabert lr 1.15e-5",
  dev={"accuracy_seed0": 0.870, "accuracy_seed1": 0.840, "baseline_accuracy": 0.740}, gpu_minutes=3, outcome="success",
  takeaway="UltraBERT handles PAWS; large seed variance (3 pts)")
L(task=P, kind="quantisation", idea="UltraBERT full dynamic int8", change="int8 all MatMuls", dev={"accuracy": 0.625, "baseline_accuracy": 0.870},
  outcome="failure", takeaway="confirms v1: int8 breaks UltraBERT")
L(task=P, kind="quantisation", idea="UltraBERT int8 excluding attention act x act MatMuls", change="49 nodes excluded", dev={"accuracy": 0.625, "baseline_accuracy": 0.870},
  outcome="failure", takeaway="attention matmuls are not the culprit")
L(task=P, kind="quantisation", idea="UltraBERT partial int8 excluding attention matmuls + all mlp.down_proj", change="73 of 217 MatMuls kept fp32",
  dev={"accuracy": 0.865, "baseline_accuracy": 0.870}, outcome="success",
  takeaway="down_proj (large-activation outliers) is what breaks UltraBERT int8; p95 46 ms vs 88 ms fp32 ONNX")
L(task=P, kind="data", idea="distillation: teacher UltraBERT pseudo-labels the 44k unlabeled pool (soft), train student on LLM+pseudo",
  change="48.7k training pairs", dev={"ub_student_s0": 0.870, "ub_student_s1": 0.870, "gte_student": 0.860, "gte_student_int8": 0.845, "baseline_teacher": 0.870},
  gpu_minutes=25, outcome="success", takeaway="pseudo labels make gte learn PAWS (0.74->0.86) and stabilise UltraBERT seeds; gte int8 loses 1.5 pts")
L(task=P, kind="data", idea="pseudo labels from 2-teacher average (s0 0.87 + s1 0.84)", change="ensemble pseudo labels", dev={"accuracy": 0.830, "baseline_accuracy": 0.870},
  gpu_minutes=6, outcome="failure", takeaway="averaging in the weaker teacher hurt")
L(task=P, kind="labeling", idea="3.2k more pool labels (flash + v2 seeds)", change="7.7k LLM labels total", cost_usd=0.11,
  dev={"teacher_s0": 0.865, "teacher_s1": 0.860, "baseline_mean_4.5k": 0.855}, gpu_minutes=8, outcome="neutral",
  takeaway="seed variance shrinks; mean +0.75 pts (within noise); 2-teacher ensemble dev 0.875")
