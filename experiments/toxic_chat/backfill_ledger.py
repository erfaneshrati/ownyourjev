import sys; sys.path.insert(0, ".")
from benchmarks.v2.ledger import log_experiment as L
T = "toxic_chat"
L(task=T, kind="labeling", idea="baseline labeler: deepseek-v4.1-flash, task spec only, 20 seeds (all negative), soft labels", change="dev labeling",
  dev={"toxic_f1": 0.774, "toxic_ap": 0.620, "jailbreak_f1": 0.533, "n": 200}, cost_usd=0.0075, outcome="neutral", takeaway="reference labeler; seeds contain 0 positives so they only teach negatives")
L(task=T, kind="labeling", idea="labeling guidance v1: ToxicChat convention (sexual requests, harmful how-to, hate jokes, jailbreak role-play count as toxic; neutral sensitive topics do not)",
  change="guidance added to prompt", dev={"toxic_f1": 0.857, "toxic_ap": 0.745, "jailbreak_f1": 0.571, "baseline_toxic_f1": 0.774, "n": 200}, cost_usd=0.0098, outcome="success",
  takeaway="explicit ToxicChat convention lifts labeler toxic F1 by +0.08")
L(task=T, kind="labeling", idea="guidance v2: + intent rule (benign overrides/personas are neither toxic nor jailbreak; vulgar slang counts)",
  change="guidance v1 -> v2", dev={"toxic_f1": 0.833, "jailbreak_f1": 0.714, "baseline_toxic_f1": 0.857, "baseline_jailbreak_f1": 0.571, "n": 200}, cost_usd=0.0095, outcome="neutral",
  takeaway="toxic -1 example (noise), jailbreak +0.14; kept v2 for its jailbreak gain")
L(task=T, kind="labeling", idea="guidance v1 without the 20 seed examples", change="drop seeds", dev={"toxic_f1": 0.857, "toxic_ap": 0.788, "jailbreak_f1": 0.571, "n": 200}, cost_usd=0.0083,
  outcome="neutral", takeaway="all-negative seeds neither help nor hurt; 3x cheaper prompt")
L(task=T, kind="labeling", idea="labeler comparison with guidance v2: claude-haiku-5.5, gemini-2.5-flash-lite, qwen3.7-flash",
  change="labeler model", dev={"haiku_toxic_f1": 0.800, "haiku_jb_f1": 0.750, "gemlite_toxic_f1": 0.812, "gemlite_jb_f1": 0.571, "qwen_toxic_f1": 0.857, "qwen_jb_f1": 0.706, "deepseek_toxic_f1": 0.833, "n": 200},
  cost_usd=0.087, outcome="neutral", takeaway="no labeler clearly better; qwen3.7-flash ties best at lowest cost; haiku 7x cost, no gain")
L(task=T, kind="labeling", idea="labeler ensemble (mean of soft labels): deepseek g2 + qwen g2", change="2-labeler average",
  dev={"toxic_bestF1": 0.882, "toxic_cvF1": 0.882, "toxic_ap": 0.786, "jailbreak_cvF1": 0.644, "baseline_toxic_f1": 0.833, "n": 200}, cost_usd=0.0, outcome="success",
  takeaway="averaging two cheap labelers gives graded labels and the best dev F1; adding haiku/gemini raised AUC (0.99) but not F1")
L(task=T, kind="labeling", idea="label full pool (4862) with deepseek g2 + qwen g2 soft labels", change="pool labeling", dev={}, cost_usd=0.29,
  outcome="neutral", takeaway="373/349 toxic (7.7%/7.2%), 317 agree; 86 borderline; labeling ran ~1 call/s due to shared OpenRouter load")
L(task=T, kind="latency", idea="gte-modernbert shared encoder, two sigmoid heads, ONNX fp32 vs dynamic int8", change="v1silver_hard model",
  dev={"fp32_p95_ms": 103.6, "int8_p95_ms": 58.0, "fp32_toxic_ap": 0.832, "int8_toxic_ap": 0.799, "int8_max_abs_prob_diff": 0.37}, outcome="neutral",
  takeaway="fp32 misses the gate; full int8 passes but drifts (toxic bestF1 0.788->0.750)")
L(task=T, kind="quantisation", idea="selective int8: keep mlp/Wo (GeGLU output projection) in fp32", change="nodes_to_exclude mlp/Wo + head",
  dev={"max_abs_prob_diff": 0.043, "toxic_ap": 0.820, "toxic_bestF1": 0.788, "p95_ms": 67.7, "full_int8_p95_ms": 58.0}, outcome="success",
  takeaway="mlp/Wo activation outliers cause the int8 drift; excluding them restores fp32 accuracy at ~+10 ms p95 (busy machine); no single layer dominates")
L(task=T, kind="training", idea="student baseline on v1 silver_seeded hard labels (shared encoder, 2 heads, gte-modernbert full FT)", change="baseline",
  dev={"toxic_f1@0.5": 0.710, "toxic_ap": 0.832, "toxic_bestF1": 0.788, "toxic_cvF1": 0.703, "jb_cvF1": 0.551}, gpu_minutes=8, outcome="neutral", takeaway="ranking good (AUC .98) but threshold-unstable; best threshold 0.1")
L(task=T, kind="training", idea="student on new 2-labeler soft labels (dsqw)", change="labels v1 hard -> ds+qwen g2 soft",
  dev={"toxic_ap": 0.823, "toxic_cvF1": 0.751, "toxic_bestF1": 0.778, "baseline_toxic_cvF1": 0.703}, gpu_minutes=10, outcome="neutral", takeaway="within dev noise; misses short crude/hateful prompts")
L(task=T, kind="data", idea="synthetic positives (374, steered by pool prompts both labelers flagged, relabeled by ensemble)", change="+synth_pos",
  dev={"toxic_f1@0.5": 0.800, "toxic_ap": 0.843, "toxic_cvF1": 0.734, "baseline_toxic_ap": 0.823}, cost_usd=0.06, gpu_minutes=10, outcome="neutral", takeaway="F1@0.5 up, AP slightly up, cvF1 within noise")
L(task=T, kind="training", idea="pos_weight 3 on both heads (+synth)", change="pos_weight 1->3", dev={"toxic_ap": 0.862, "toxic_cvF1": 0.718, "baseline_toxic_cvF1": 0.734}, gpu_minutes=10, outcome="neutral", takeaway="no clear gain")
L(task=T, kind="training", idea="student on v1 silver plain+seeded averaged (soft)", change="v1 two-label ensemble", dev={"toxic_ap": 0.877, "toxic_cvF1": 0.723, "toxic_bestF1": 0.800}, gpu_minutes=10, outcome="neutral", takeaway="best AP but threshold low (0.05)")
L(task=T, kind="data", idea="+287 short crude/hateful synthetic positives +376 hard negatives (benign personas, benign overrides, sensitive educational)", change="+synth_pos2 +synth_neg",
  dev={"toxic_ap": 0.854, "toxic_cvF1": 0.754, "toxic_bestF1": 0.800, "baseline_toxic_cvF1": 0.734}, cost_usd=0.05, gpu_minutes=10, outcome="neutral", takeaway="small, within-noise improvement")
L(task=T, kind="other", idea="GPU scheduler deadlock killed two runs (v1ens, dsqw); OOM on shared GPUs", change="added grad checkpointing, bs8x2 accumulation, OOM retry", outcome="failure",
  takeaway="infrastructure; reran in new processes")
print("ok")
