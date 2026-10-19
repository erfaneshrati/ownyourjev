"""Backfill of the summeval experiment ledger (experiments run before the ledger existed). Dev = 204 gold, 14 docs,
group-Spearman per dim, mean over 4 dims. Student runs: fold-0 held out of training, dev docs never trained on."""
from benchmarks.v2.ledger import log_experiment as L
T = "summeval"
B = dict(task=T, round=1, backfilled=True)
# ---- labeler / prompt selection (LLM labels vs dev gold) ----
L(**B, kind="labeling", idea="labeler comparison with G-Eval-style rubric guidance (g1: rich criteria + per-level anchors, soft labels)",
  change="deepseek-v4.1-flash / claude-haiku-5.5 / gemini-2.5-flash-lite / qwen3.7-flash, all with g1",
  dev={"mean": {"deepseek": 0.466, "haiku": 0.456, "gemini-2.5-flash-lite": 0.412, "qwen3.7-flash": 0.557},
       "qwen_per_dim": {"coherence": 0.638, "consistency": 0.451, "fluency": 0.548, "relevance": 0.590}, "n": 204},
  cost_usd=0.184, outcome="success", takeaway="qwen3.7-flash is both the cheapest and the best labeler on dev (0.557)")
L(**B, kind="labeling", idea="rubric guidance vs plain task prompt (qwen)", change="plain spec questions vs g1 rubric",
  dev={"mean": 0.423, "baseline_mean_g1": 0.557, "n": 204}, cost_usd=0.013, outcome="success",
  takeaway="the rubric (definitions, per-level anchors, 'ignore tokenisation artefacts') is worth +0.13 dev")
L(**B, kind="labeling", idea="20 gold seeds as few-shot examples in the labeling prompt",
  change="g1 + seeds (qwen, deepseek)", dev={"mean_qwen": 0.477, "baseline_qwen": 0.557, "mean_deepseek": 0.498, "baseline_deepseek": 0.466, "n": 204},
  cost_usd=0.133, outcome="failure", takeaway="seeds hurt qwen (-0.08), helped deepseek a bit; seeds share docs with dev so even this is optimistic")
L(**B, kind="labeling", idea="CoT: short 'analysis' field before the scores (qwen)", change="g1_cot",
  dev={"mean": 0.468, "baseline_mean": 0.557, "n": 204}, cost_usd=0.011, outcome="failure",
  takeaway="alone worse than greedy g1, but useful as an extra vote in a self-consistency ensemble")
L(**B, kind="labeling", idea="self-consistency: average soft labels of several qwen samples (T=0.7) + greedy",
  change="x3 = greedy+2 samples; x5 = greedy+3 samples+CoT", dev={"x1": 0.557, "single_sample_runs": [0.498, 0.496, 0.491], "x3": 0.580, "x5": 0.599, "n": 204},
  cost_usd=0.068, outcome="success", takeaway="teacher improves +0.02-0.04 with votes; applied to the 1376 pool texts")
L(**B, kind="labeling", idea="cross-model label ensembles (qwen x3 + haiku / deepseek / gemini)",
  change="averaged expected scores", dev={"qwen_x3": 0.580, "+haiku": 0.594, "+deepseek": 0.577, "+gemini": 0.581, "n": 204},
  cost_usd=0.0, outcome="neutral", takeaway="+haiku +0.014 (noise level), not worth $0.5 on the pool")
# ---- latency / architecture ----
L(**B, kind="latency", idea="CPU latency probe of gte-modernbert int8 ONNX vs layers and sequence length (stock HF export, sdpa)",
  change="22/12 layers x 256-768 tokens", dev={"p95_ms": {"22L_256": 142, "22L_512": 600, "22L_768": 1437, "12L_256": 77}},
  outcome="neutral", takeaway="full inputs (~560-790 tokens) are 10-20x over the gate; stock export spends most time in Where/IsNaN mask ops")
L(**B, kind="latency", idea="eager attention export instead of sdpa", change="attn_implementation=eager",
  dev={"p95_ms": {"12L_256_sdpa": 77, "12L_256_eager": 59, "22L_128": 53, "22L_192": 89, "16L_192": 56, "16L_256": 82}},
  outcome="success", takeaway="removes the O(L^2) mask ops; budget is ~16 layers x 192 tokens or 22 x 128")
L(**B, kind="latency", idea="export-friendly ModernBERT forward (precomputed rotary tables, rotate_half as constant matmul, one window mask)",
  change="experiments/summeval/fastmb.py; bit-identical outputs", dev={"p95_ms_16L_192": {"eager": "56-70", "fast": "53-56"}},
  outcome="success", takeaway="~10-20% faster and more stable p95 under load")
# ---- student architecture / input ----
L(**B, kind="training", idea="layer-truncated gte-modernbert x input length (shared encoder, 4 heads, lexical features, soft labels, qwen g1 labels)",
  change="22L/512, 22L/160, 12L/256, 16L/192 at 4 epochs", dev={"mean": {"22L_512": 0.437, "22L_160": 0.444, "12L_256": 0.482, "16L_192": 0.500}, "n": 204},
  gpu_minutes=5, outcome="success", takeaway="long article context does not help the student; 16 layers x 192 tokens fits the gate and is best")
L(**B, kind="training", idea="more epochs (8 vs 4-6)", change="epochs", dev={"e8_seeds": [0.504, 0.470], "e6_curve_peak": "ep5-6"},
  gpu_minutes=4, outcome="neutral", takeaway="dev peaks at epoch 5-6; 6 epochs fixed")
L(**B, kind="training", idea="within-article pairwise ranking loss on expected scores", change="rank_w=1.0",
  dev={"mean": 0.485, "baseline_mean_same_seed": 0.504}, gpu_minutes=2, outcome="failure", takeaway="no gain over soft-label KL")
L(**B, kind="training", idea="drop the lexical alignment features (full-article n-gram novelty, lead coverage, repetition, truncation)",
  change="use_feats=False", dev={"mean": 0.485, "baseline_mean_same_seed": 0.504}, gpu_minutes=2, outcome="neutral",
  takeaway="features are within noise but kept (cheap, give the model full-article signal)")
L(**B, kind="training", idea="lr 5e-5 vs 3e-5", change="lr", dev={"mean_lr5e-5": 0.501, "baseline": 0.516}, gpu_minutes=2, outcome="neutral", takeaway="no gain")
L(**B, kind="labeling", idea="train the student on self-consistency labels (x3) instead of greedy x1", change="lab_eval__g1x3",
  dev={"mean_3seeds": 0.503, "baseline_x1_3seeds_approx": 0.498}, cost_usd=0.227, gpu_minutes=4, outcome="neutral",
  takeaway="better teacher (+0.02 on dev) transfers only marginally to the student")
L(**B, kind="data", idea="aligned context: summary + article sentences chosen by lexical overlap with the summary (+ lead sentence) instead of the article lead",
  change="ctx=align", dev={"mean_3seeds": 0.509, "baseline_lead_3seeds": 0.503, "consistency": "0.40 vs 0.39"}, gpu_minutes=4, outcome="neutral",
  takeaway="small consistent gain on consistency/relevance; kept")
# ---- synthetic data ----
L(**B, kind="data", idea="LLM-generated summaries with controlled quality profiles (18 profiles: hallucination, disfluent, incoherent order, redundant, truncated, off-focus, extractive variants...), 9 per article call, PTB/lowercase style mimicry, labeled by qwen g1",
  change="+650 synthetic (training-fold articles) to 1087 real", dev={"mean_2seeds": 0.445, "baseline_2seeds": 0.487, "consistency": "0.25 vs 0.33"},
  cost_usd=0.093, gpu_minutes=5, outcome="failure",
  takeaway="synthetic flaws are much grosser than the real system-vs-system differences; student shifts to novelty cues and loses within-article ranking")
L(**B, kind="data", idea="synthetic variants: weight 0.3 / exclude consistency head / drop hallucination profiles",
  change="syn_w=0.3, syn_dims, syn_drop", dev={"syn_w0.3": 0.453, "no_consistency": 0.448, "drop_halluc": 0.469, "baseline": 0.487},
  gpu_minutes=5, outcome="failure", takeaway="every mix of LLM-written synthetic summaries is below real-only")
L(**B, kind="data", idea="programmatic extractive summaries in SummEval-system style (lead-3, lead-2, random/late sentences, duplicated sentence, length-capped lead), LLM-labeled",
  change="+560 synthetic to real (align ctx, x3 labels)", dev={"mean_seeds": [0.460, 0.487], "syn_w0.5": 0.474, "baseline_3seeds": 0.509},
  cost_usd=0.062, gpu_minutes=5, outcome="failure", takeaway="even in-distribution-looking synthetic summaries hurt")
L(**B, kind="data", idea="LLM-written CNN/DM-style new articles (281) as a source for synthetic summaries",
  change="generated only", cost_usd=0.036, outcome="neutral", takeaway="not used: synthetic summaries failed on real articles first")
L(**B, kind="data", idea="add dev gold (one half of dev docs) to training, evaluate on the other half",
  change="--dev_half h, gold_w 1 or 3", dev={"half0_eval": {"w1": 0.476, "w3": 0.495, "baseline_3seeds": [0.579, 0.542, 0.539]}, "half1_eval": {"w1": 0.401, "w3": 0.438, "baseline_3seeds": [0.473, 0.477, 0.446]}},
  gpu_minutes=5, outcome="failure", takeaway="mixing 110 gold (sharp, mostly-4 targets) into LLM soft labels hurts; gold not used for training")
L(**B, kind="data", idea="add LLM (x3) labels of dev-half texts (more real-distribution data), evaluate on the other half",
  change="--dev_half h --dev_llm", dev={"half0_eval": 0.566, "half1_eval": 0.457, "baseline_mean": [0.553, 0.465]}, gpu_minutes=3,
  outcome="neutral", takeaway="+100 real texts is neutral: data volume is not the bottleneck")
L(**B, kind="other", idea="student-teacher fidelity on held-out fold 0 (group Spearman student vs qwen x3 labels)",
  change="analysis", dev={"coherence": 0.689, "consistency": 0.547, "fluency": 0.599, "relevance": 0.640}, outcome="neutral",
  takeaway="the 16L/192 student reproduces only ~0.6-0.7 of the teacher's within-article ranking")
L(**B, kind="training", idea="weight soup of 3 seeds", change="uniform average of fine-tuned weights",
  dev={"soup": 0.486, "members": [0.523, 0.510, 0.488], "prediction_ensemble": 0.528}, gpu_minutes=5, outcome="failure",
  takeaway="seeds land in different basins (different head init); prediction ensembles help (+0.02) but cost latency")
L(**B, kind="labeling", idea="x5 labels (greedy + 3 samples + CoT) for the student", change="lab_eval__g1x5",
  dev={"mean_3seeds": 0.504, "baseline_x3_3seeds": 0.509}, cost_usd=0.287, gpu_minutes=4, outcome="neutral", takeaway="teacher x5 is better on dev but the student does not gain")
L(**B, kind="training", idea="pool CLS + masked mean instead of CLS", change="pool=clsmean (x5 labels)",
  dev={"mean_3seeds": 0.5145, "per_seed": [0.521, 0.511, 0.511], "baseline_cls_3seeds": 0.504}, gpu_minutes=4, outcome="success",
  takeaway="+0.01 and lower seed variance; kept")
print("ok")
