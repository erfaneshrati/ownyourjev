"""Backfill the experiment ledger (round 1) for ag_news, financial_phrasebank, emotion from NOTES.md dev numbers.
No new experiments. gpu_minutes = wall-clock train time on a shared GPU (upper bound)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.v2.ledger import log_experiment as L

def gm(s): return round(s / 60, 1)

# ---------------- ag_news ----------------
T = "ag_news"; N = 200
L(task=T, kind="labeling", idea="labeler deepseek-v4.1-flash, plain v1 prompt, on dev", change="labeled 200 dev",
  dev={"labeler_agreement": 0.905, "A": 0.900, "B": 0.910, "n": N}, cost_usd=0.009, outcome="success",
  takeaway="plain labels already ~0.905 agreement with gold; v1 plain pool labels reusable for free")
L(task=T, kind="labeling", idea="add 20 v2 gold seeds as few-shot examples", change="seeds in labeling prompt",
  dev={"labeler_agreement": 0.905, "A": 0.880, "B": 0.930, "baseline_labeler_agreement": 0.905, "n": N}, cost_usd=0.0061,
  outcome="neutral", takeaway="seeds do not change labeler agreement on AG News")
L(task=T, kind="labeling", idea="seeds + soft label distribution", change="distribution=True",
  dev={"labeler_agreement": 0.910, "A": 0.900, "B": 0.920, "baseline_labeler_agreement": 0.905, "n": N}, cost_usd=0.0123,
  outcome="neutral", takeaway="+0.5 pt is within noise; not worth buying new pool labels")
L(task=T, kind="training", idea="gte-modernbert full FT on reused v1 plain labels (4,970 texts)", change="baseline student, 3 epochs, max_len 128",
  dev={"accuracy": 0.920, "A": 0.91, "B": 0.93, "n": N}, gpu_minutes=gm(135), outcome="success",
  takeaway="student exceeds its labeler (0.905) on dev; chosen recipe")
L(task=T, kind="data", idea="learning curve: 2,500 v1 plain labels instead of 4,970", change="--n 2500",
  dev={"accuracy": 0.925, "baseline_accuracy": 0.920, "n": N}, gpu_minutes=gm(383), outcome="neutral",
  takeaway="flat learning curve 2.5k->5k, so no extra labels from the 120k pool were bought")
L(task=T, kind="labeling", idea="train on v1 seeded labels (v1 pool-drawn seeds) instead of plain", change="v1:silver_seeded",
  dev={"accuracy": 0.900, "baseline_accuracy": 0.920, "n": N}, gpu_minutes=gm(269), outcome="failure",
  takeaway="v1 seeded labels give a slightly worse student; plain kept")
L(task=T, kind="quantisation", idea="ONNX dynamic int8 (per-channel) of the chosen model", change="export + quantize_int8",
  dev={"accuracy": 0.915, "baseline_accuracy": 0.920, "argmax_agree_fp32": 0.975, "cpu_p95_ms": 50.8, "n": N}, outcome="success",
  takeaway="int8 costs 0.5 pt and gives p95 47-51 ms on the busy machine; frozen")
L(task=T, kind="selection", idea="freeze: v1 plain labels + gte-modernbert + int8", change="FROZEN.json",
  dev={"accuracy": 0.915, "n": N}, outcome="success", takeaway="selected on dev; scored on test once")

# ---------------- financial_phrasebank ----------------
T = "financial_phrasebank"
L(task=T, kind="labeling", idea="labeler deepseek-v4.1-flash, plain v1 prompt, on dev (cached v1 calls)", change="labeled 200 dev",
  dev={"labeler_agreement": 0.850, "A": 0.810, "B": 0.890, "n": N}, cost_usd=0.0, outcome="success",
  takeaway="v1 plain labels cover ~all of the pool and agree 0.85 with dev gold")
L(task=T, kind="labeling", idea="add 20 v2 gold seeds", change="seeds in prompt",
  dev={"labeler_agreement": 0.845, "A": 0.820, "B": 0.870, "baseline_labeler_agreement": 0.850, "n": N}, cost_usd=0.0043,
  outcome="neutral", takeaway="seeds do not help the labeler")
L(task=T, kind="labeling", idea="seeds + soft label distribution", change="distribution=True",
  dev={"labeler_agreement": 0.820, "A": 0.800, "B": 0.840, "baseline_labeler_agreement": 0.850, "n": N}, cost_usd=0.0086,
  outcome="failure", takeaway="soft labels lower argmax agreement")
L(task=T, kind="labeling", idea="guidance G1: positive-leaning investor rules from dev-half-A errors (gold positive -> labeled neutral)",
  change="guidance, no seeds", dev={"labeler_agreement": 0.845, "A": 0.880, "B": 0.810, "baseline_labeler_agreement": 0.850, "baseline_B": 0.890, "n": N},
  cost_usd=0.0049, outcome="failure", takeaway="fixes half A but hurts held-out half B: overfits the dev errors")
L(task=T, kind="labeling", idea="guidance G1 + 20 v2 seeds", change="guidance + seeds",
  dev={"labeler_agreement": 0.850, "A": 0.860, "B": 0.840, "baseline_labeler_agreement": 0.850, "baseline_B": 0.890, "n": N},
  cost_usd=0.0047, outcome="failure", takeaway="no gain on held-out half; no new pool labels bought")
L(task=T, kind="training", idea="student on reused v1 plain labels (3,637 texts), 2 seeds", change="gte-modernbert full FT, max_len 96",
  dev={"accuracy_seed0": 0.840, "accuracy_seed1": 0.850, "mean_accuracy": 0.845, "n": N}, gpu_minutes=gm(134 + 41), outcome="neutral",
  takeaway="baseline student ~ labeler level")
L(task=T, kind="labeling", idea="student on reused v1 seeded labels (v1 pool-drawn seeds), 2 seeds", change="v1:silver_seeded",
  dev={"accuracy_seed0": 0.850, "accuracy_seed1": 0.860, "mean_accuracy": 0.855, "baseline_mean_accuracy": 0.845, "n": N},
  gpu_minutes=gm(50 + 43), outcome="success", takeaway="best mean (within noise); chosen")
L(task=T, kind="data", idea="agreement filter: keep texts where v1 plain and v1 seeded labels agree (3,221)", change="drop 416 disagreements",
  dev={"accuracy_seed0": 0.865, "accuracy_seed1": 0.845, "mean_accuracy": 0.855, "baseline_mean_accuracy": 0.855, "n": N},
  gpu_minutes=gm(226 + 39), outcome="neutral", takeaway="same mean as seeded labels, higher seed variance; not used")
L(task=T, kind="quantisation", idea="ONNX int8 variants of the seed-1 seeded-label model",
  change="per-channel s8 / u8 / per-tensor s8 / reduce_range vs fp32 ONNX",
  dev={"fp32_onnx": 0.860, "int8_pc_s8": 0.850, "int8_pc_u8": 0.820, "int8_per_tensor": 0.805, "int8_reduce_range": 0.830,
       "argmax_agree_fp32_pc_s8": 0.85, "n": N}, outcome="neutral",
  takeaway="int8 flips ~15% of borderline neutral/positive decisions; per-channel s8 is the least harmful")
L(task=T, kind="latency", idea="serve fp32 ONNX to avoid int8 loss", change="fp32 ONNX latency, 4 threads",
  dev={"accuracy": 0.860, "cpu_p95_ms_run1": 56.6, "cpu_p95_ms_run2": 69.1}, outcome="failure",
  takeaway="fp32 p95 too close to the 70 ms gate on the shared machine; int8 (p95 32 ms) frozen instead")
L(task=T, kind="selection", idea="freeze: v1 seeded labels + gte-modernbert seed 1 + int8", change="FROZEN.json",
  dev={"accuracy": 0.850, "n": N}, outcome="success", takeaway="selected on dev; scored on test once")

# ---------------- emotion ----------------
T = "emotion"
L(task=T, kind="labeling", idea="labeler deepseek-v4.1-flash, plain v1 prompt", change="labeled 200 dev",
  dev={"labeler_agreement": 0.490, "A": 0.450, "B": 0.530, "n": N}, cost_usd=0.0046, outcome="failure",
  takeaway="plain labeler judges overall mood; caps near 0.5")
L(task=T, kind="labeling", idea="add 20 v2 gold seeds", change="seeds in prompt",
  dev={"labeler_agreement": 0.540, "A": 0.480, "B": 0.600, "baseline_labeler_agreement": 0.490, "n": N}, cost_usd=0.0073,
  outcome="success", takeaway="+5 pts, but still far from usable")
L(task=T, kind="labeling", idea="seeds + soft label distribution", change="distribution=True",
  dev={"labeler_agreement": 0.510, "A": 0.520, "B": 0.500, "baseline_labeler_agreement": 0.540, "n": N}, cost_usd=0.0123,
  outcome="failure", takeaway="soft labels lower agreement and double output cost")
L(task=T, kind="labeling", idea="guidance G1 (label = category of the word attached to feel/feeling, ignore negation/context, word->category list), no seeds",
  change="guidance from seeds + dev-half-A errors", dev={"labeler_agreement": 0.765, "A": 0.830, "B": 0.700, "baseline_labeler_agreement": 0.490, "baseline_B": 0.530, "n": N},
  cost_usd=0.0063, outcome="success", takeaway="the dataset convention is the feel-word; +17 pts on the held-out half")
L(task=T, kind="labeling", idea="guidance G1 + 20 v2 seeds", change="guidance + seeds",
  dev={"labeler_agreement": 0.785, "A": 0.840, "B": 0.730, "baseline_labeler_agreement": 0.540, "baseline_B": 0.600, "n": N},
  cost_usd=0.0067, outcome="success", takeaway="best affordable labeler; used for pool labels")
L(task=T, kind="labeling", idea="G1 + seeds with qwen/qwen3.7-flash", change="labeler model",
  dev={"labeler_agreement": 0.745, "A": 0.790, "B": 0.700, "baseline_labeler_agreement": 0.785, "n": N}, cost_usd=0.0049,
  outcome="failure", takeaway="cheaper but 4 pts worse")
L(task=T, kind="labeling", idea="G1 + seeds with anthropic/claude-haiku-5.5", change="labeler model",
  dev={"labeler_agreement": 0.805, "A": 0.850, "B": 0.760, "baseline_labeler_agreement": 0.785, "n": N}, cost_usd=0.05,
  outcome="neutral", takeaway="+2-3 pts but ~7.5x cost (~$2.5 for 10k); not affordable in the shared cap")
L(task=T, kind="training", idea="student on reused v1 plain labels (5k)", change="gte-modernbert full FT, max_len 64",
  dev={"accuracy": 0.535, "n": N}, gpu_minutes=gm(80), outcome="failure", takeaway="v1 plain labels are too noisy for this dataset")
L(task=T, kind="training", idea="student on reused v1 seeded labels (5k, v1 pool-drawn seeds)", change="v1:silver_seeded",
  dev={"accuracy": 0.600, "baseline_accuracy": 0.535, "n": N}, gpu_minutes=gm(60), outcome="failure",
  takeaway="v1 seeded labels still cap the student near 0.6")
L(task=T, kind="labeling", idea="label 5,000 pool texts with G1 + seeds (v1-labeled uids first)", change="new pool labels",
  dev={}, cost_usd=0.1437, outcome="success", takeaway="agrees 0.69 with v1 seeded labels on shared uids; class mix sadness/joy-heavy like gold")
L(task=T, kind="training", idea="student on G1+seeds labels, 5k", change="max_len 96, 3 epochs",
  dev={"accuracy": 0.795, "A": 0.81, "B": 0.78, "baseline_accuracy": 0.600, "n": N}, gpu_minutes=gm(227), outcome="success",
  takeaway="+19.5 pts over v1 labels: guidance is the lever")
L(task=T, kind="training", idea="same, seed 1", change="--seed 1",
  dev={"accuracy": 0.780, "baseline_accuracy": 0.795, "n": N}, gpu_minutes=gm(2124), outcome="neutral", takeaway="seed variance ~1.5 pts")
L(task=T, kind="training", idea="same, 5 epochs", change="--epochs 5",
  dev={"accuracy": 0.795, "baseline_accuracy": 0.795, "n": N}, gpu_minutes=gm(446), outcome="neutral",
  takeaway="early stopping picks epoch 2 either way")
L(task=T, kind="data", idea="learning curve: 2,500 G1 labels", change="--n 2500",
  dev={"accuracy": 0.735, "baseline_accuracy": 0.795, "n": N}, gpu_minutes=gm(1324), outcome="success",
  takeaway="steep curve 2.5k->5k justified buying 5k more labels")
L(task=T, kind="labeling", idea="label 5,000 more pool texts with G1 + seeds (10k total)", change="new pool labels",
  dev={}, cost_usd=0.1435, outcome="success", takeaway="pool labels for the 10k run")
L(task=T, kind="data", idea="student on 10k G1 labels, 2 seeds", change="10k instead of 5k",
  dev={"accuracy_seed0": 0.800, "accuracy_seed1": 0.790, "baseline_accuracy_seed0": 0.795, "baseline_accuracy_seed1": 0.780, "n": N},
  gpu_minutes=gm(1371 + 1636), outcome="neutral", takeaway="+0.5-1 pt: student is at the labeler ceiling; chosen (seed 0)")
L(task=T, kind="quantisation", idea="ONNX dynamic int8 (per-channel) of the 5k and 10k models", change="quantize_int8",
  dev={"accuracy_int8_5k": 0.780, "accuracy_fp32_5k": 0.795, "accuracy_int8_10k": 0.780, "accuracy_fp32_10k": 0.800,
       "argmax_agree_fp32": 0.935, "cpu_p95_ms_int8": 22.2, "n": N}, outcome="failure",
  takeaway="int8 costs 1.5-2 pts on emotion; not used")
L(task=T, kind="latency", idea="serve fp32 ONNX", change="fp32 ONNX latency, 4 threads",
  dev={"cpu_p95_ms_5k": 45.5, "cpu_p95_ms_10k": 43.6, "accuracy": 0.800}, outcome="success",
  takeaway="short tweets keep fp32 well under the 70 ms gate")
L(task=T, kind="selection", idea="freeze: G1+seeds 10k labels + gte-modernbert seed 0 + fp32 ONNX", change="FROZEN.json",
  dev={"accuracy": 0.800, "n": N}, outcome="success", takeaway="selected on dev; scored on test once")
