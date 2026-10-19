from benchmarks.v2.ledger import log_experiment as L
T = dict(task="boolq", round=1)
L(**T, kind="labeling", idea="labeler comparison on dev (plain prompt)", change="deepseek-v4.1-flash plain",
  dev={"agreement": 0.865, "n": 200}, cost_usd=0.0045, outcome="neutral", takeaway="cheap but weakest; seeds needed")
L(**T, kind="labeling", idea="20 gold seeds in labeling prompt", change="deepseek-v4.1-flash + seeds",
  dev={"agreement": 0.93, "baseline_agreement": 0.865, "n": 200}, cost_usd=0.017, outcome="success", takeaway="+6.5 pts but ~4x cost per call")
L(**T, kind="labeling", idea="labeler comparison", change="qwen3.7-flash plain", dev={"agreement": 0.925, "n": 200}, cost_usd=0.002,
  outcome="success", takeaway="best cost/quality labeler; chosen for full pool")
L(**T, kind="labeling", idea="seeds for qwen", change="qwen3.7-flash + seeds", dev={"agreement": 0.92, "baseline_agreement": 0.925, "n": 200},
  cost_usd=0.0206, outcome="failure", takeaway="no gain at 10x cost")
L(**T, kind="labeling", idea="soft labels from qwen", change="qwen3.7-flash distribution=True", dev={"agreement": 0.925, "n": 200},
  cost_usd=0.0022, outcome="neutral", takeaway="same accuracy; probabilities nearly binary")
L(**T, kind="labeling", idea="labeler comparison", change="gemini-3.1-flash-lite plain", dev={"agreement": 0.93, "n": 200}, cost_usd=0.0136,
  outcome="success", takeaway="strong; used as tie-breaker")
L(**T, kind="labeling", idea="labeler comparison", change="claude-haiku-5.5 plain", dev={"agreement": 0.865, "n": 200}, cost_usd=0.0083,
  outcome="failure", takeaway="weak on BoolQ")
L(**T, kind="labeling", idea="stronger labeler", change="deepseek-v4-pro plain", dev={"agreement": 0.895, "n": 200}, cost_usd=0.0967,
  outcome="failure", takeaway="expensive and worse than qwen flash")
L(**T, kind="labeling", idea="3-labeler majority", change="majority(flash+seeds, qwen, gemini) on dev", dev={"agreement": 0.95, "n": 200},
  cost_usd=0.0, outcome="success", takeaway="ensemble labels best; labels not the bottleneck")
L(**T, kind="data", idea="LLM-generated questions per passage (8/passage, balanced)", change="qwen generator vs deepseek generator, 100-passage pilot, verified by gemini",
  dev={"gen_label_agreement_qwen": 0.99, "gen_label_agreement_deepseek": 0.965, "n": 200}, cost_usd=0.062, outcome="success",
  takeaway="qwen-generated answers reliable and 3.5x cheaper")
L(**T, kind="latency", idea="lean ONNX re-implementation of encoder (masks once, RoPE tables, split fused weights)", change="vs stock HF export, gte 12L int8 @256 tok",
  dev={"p50_ms_before": 72, "p50_ms_after": 40}, outcome="success", takeaway="stock export spent >50% CPU in mask plumbing; ~1.8x speedup, identical outputs")
L(**T, kind="training", idea="baseline gte-modernbert 22L on v1 seeded labels", change="4.5k v1 labels", dev={"accuracy": 0.74, "auroc": 0.80, "n": 200},
  gpu_minutes=5, outcome="neutral", takeaway="weak baseline; data-hungry")
L(**T, kind="training", idea="mean pooling", change="gte, mean vs cls", dev={"accuracy": 0.69, "baseline_accuracy": 0.74, "n": 200}, gpu_minutes=4,
  outcome="failure", takeaway="cls better")
L(**T, kind="data", idea="full pool with qwen labels", change="gte, 8.8k qwen labels vs 4.5k v1", dev={"accuracy": 0.835, "auroc": 0.885, "baseline_accuracy": 0.74, "n": 200},
  gpu_minutes=8, outcome="success", takeaway="more real passages + better labels: +9.5 pts")
L(**T, kind="data", idea="add 23.7k generated questions", change="gte, pool + gen", dev={"accuracy": 0.775, "auroc": 0.865, "baseline_accuracy": 0.835, "n": 200},
  cost_usd=0.167, gpu_minutes=18, outcome="failure", takeaway="passage-driven synthetic questions hurt gte")
L(**T, kind="data", idea="combined labels (qwen + v1 seeded + gemini tie-break)", change="gte, comb labels", dev={"accuracy": 0.77, "auroc": 0.878, "baseline_accuracy": 0.835, "n": 200},
  cost_usd=0.0505, gpu_minutes=8, outcome="failure", takeaway="within seed noise; no gain over qwen alone")
L(**T, kind="data", idea="generated questions only", change="gte trained on gen only", dev={"accuracy": 0.595, "auroc": 0.74, "n": 200}, gpu_minutes=12,
  outcome="failure", takeaway="synthetic questions do not transfer to real BoolQ")
L(**T, kind="training", idea="UltraBERT backbone", change="ultrabert lr1e-5, pool+gen", dev={"accuracy": 0.845, "auroc": 0.914, "baseline_accuracy": 0.775, "n": 200},
  gpu_minutes=30, outcome="success", takeaway="UltraBERT >> gte for reading comprehension")
L(**T, kind="quantisation", idea="plain dynamic int8 on UltraBERT", change="all MatMuls int8", dev={"accuracy": 0.515, "baseline_accuracy": 0.85, "n": 200},
  outcome="failure", takeaway="collapses; cause is MLP down_proj activation outliers")
L(**T, kind="quantisation", idea="keep down_proj fp32", change="int8 except down_proj", dev={"accuracy": 0.85, "agree_fp32": 0.97, "n": 200},
  outcome="success", takeaway="isolates the culprit layer")
L(**T, kind="quantisation", idea="SmoothQuant on MLP down-projection (alpha 0.8) + weight-only-matmul int8", change="UltraBERT and gte",
  dev={"accuracy_ub": 0.84, "agree_fp32_ub": 0.97, "accuracy_gte": 0.74, "baseline_gte_int8": 0.70, "n": 200}, outcome="success",
  takeaway="all weights int8 with no dev loss on both backbones")
L(**T, kind="latency", idea="max_len 224 with left truncation (keep question)", change="vs 384", dev={"accuracy_left224": 0.855, "accuracy_384": 0.85, "accuracy_right256": 0.835, "p95_ms": 65},
  outcome="success", takeaway="passes gate without accuracy loss; right truncation cuts the question")
L(**T, kind="latency", idea="RoPE via rotation matmul vs slice/concat", change="UltraBERT int8", dev={}, outcome="neutral", takeaway="no clear difference")
L(**T, kind="training", idea="UltraBERT on qwen pool labels", change="r4_ub_qwen", dev={"accuracy": 0.845, "auroc": 0.935, "n": 200}, gpu_minutes=12,
  outcome="success", takeaway="best AUROC single model")
L(**T, kind="training", idea="seed variance", change="r5_ub_qwen_s1 same recipe seed 1", dev={"accuracy": 0.82, "auroc": 0.899, "baseline_accuracy": 0.845, "n": 200},
  gpu_minutes=12, outcome="neutral", takeaway="seed variance 2-3 pts exceeds most data effects")
L(**T, kind="data", idea="combined labels on UltraBERT", change="r4_ub_comb", dev={"accuracy": 0.83, "auroc": 0.909, "baseline_accuracy": 0.845, "n": 200},
  gpu_minutes=12, outcome="neutral", takeaway="within noise")
L(**T, kind="data", idea="comb labels + generated questions on UltraBERT", change="r4_ub_comb_gen", dev={"accuracy": 0.85, "auroc": 0.917, "baseline_accuracy": 0.83, "n": 200},
  gpu_minutes=40, outcome="neutral", takeaway="gen data neutral for UltraBERT")
L(**T, kind="training", idea="curriculum: gen-pretrained then pool only", change="r6_cur", dev={"accuracy": 0.85, "auroc": 0.914, "n": 200}, gpu_minutes=10,
  outcome="neutral", takeaway="no gain")
L(**T, kind="latency", idea="drop 4 sliding layers (20L)", change="r5_ub_qwen_drop4", dev={"accuracy": 0.82, "auroc": 0.881, "baseline_auroc": 0.935, "n": 200},
  gpu_minutes=10, outcome="failure", takeaway="costs 2-3 AUROC pts; not needed with 224 left truncation")
L(**T, kind="data", idea="contrast/paraphrase augmentation of real questions (pilot)", change="100 pool items, qwen", dev={}, cost_usd=0.0037,
  outcome="failure", takeaway="flipped answers visibly noisy; not scaled")
L(**T, kind="selection", idea="prediction ensemble", change="avg probs of 4 models", dev={"accuracy": 0.86, "auroc": 0.932, "n": 200}, outcome="neutral",
  takeaway="~1 pt gain but 4x latency")
L(**T, kind="selection", idea="uniform weight soup of 6 UltraBERT fine-tunes", change="soup_all6 (frozen round 1)", dev={"accuracy_torch": 0.845, "accuracy_int8": 0.85, "auroc": 0.925, "n": 200},
  outcome="success", takeaway="free at inference; reduces seed variance; frozen recipe")
L(**T, kind="selection", idea="dev-tuned threshold", change="tune on half A, test on B and vice versa", dev={"B_at_tA": 0.81, "B_at_0.5": 0.85}, outcome="failure",
  takeaway="thresholds do not transfer; keep 0.5")
