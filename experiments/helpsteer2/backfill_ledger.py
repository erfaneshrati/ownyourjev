from benchmarks.v2.ledger import log_experiment as L
T = "helpsteer2"
def d(**k): return {**k, "n": 200, "metric": "mean Spearman over 5 attributes (gold dev)"}
L(task=T, kind="labeling", idea="labeler/prompt variants (labeler-level agreement with gold dev)",
  change="deepseek-v4.1-flash hard vs soft (distribution) vs soft+20 seeds vs soft+seeds+HelpSteer2 rubric guidance; qwen3.7-flash and gemini-2.5-flash-lite soft+seeds+guide",
  dev=d(flash_hard=0.398, flash_soft=0.419, flash_soft_seeds=0.430, flash_soft_seeds_guide=0.408, qwen_soft_seeds_guide=0.364, gemlite_soft_seeds_guide=0.204),
  cost_usd=0.34, outcome="neutral",
  takeaway="all labelers ~0.40 on dev; soft/seeds +0.02-0.03 (within noise, dev halves differ by up to 0.3 on helpfulness); rubric guidance did not help; gemini-lite poor; soft output ~2x cost so unaffordable for the pool")
L(task=T, kind="labeling", idea="compact decimal-rating labeler (one fractional 0-4 score per attribute, cheap output)",
  change="custom prompt (label_dec.py); flash and qwen, with/without seeds",
  dev=d(dec_flash=0.403, dec_qwen=0.379, dec_qwen_seeds=0.425, baseline_flash_hard=0.398), cost_usd=0.066, outcome="success",
  takeaway="qwen decimal no-seed costs 4.5e-5/call (1/3 of flash hard), labeler-level 0.379; seeds help qwen but cost 3x")
L(task=T, kind="labeling", idea="labeler ensembles (average of expected scores), labeler level on dev",
  change="flash_hard+dec_qwen; flash_hard+dec_qwen+dec_qwen_seeds",
  dev=d(flash_hard_plus_dec_qwen=0.435, plus_dec_qwen_seeds=0.463, baseline_flash_hard=0.398), outcome="success",
  takeaway="ensembling two cheap labelers +0.04 at labeler level; but see student results: did NOT transfer to the student")
L(task=T, kind="latency", idea="CPU latency of gte-modernbert int8 ONNX vs sequence length and depth (untrained, batch 1, 4 threads, loaded machine)",
  change="22/11/8/6 layers x 128-512 tokens",
  dev={"L22_128_p95": 59.7, "L22_256_p95": 139.3, "L22_512_p95": 548.3, "L11_192_p95": 48.2, "L8_256_p95": 55.0, "L6_320_p95": 59.4},
  outcome="neutral", takeaway="full inputs (median 430, p95 ~1000 tokens) are 10x over the gate; need ~128 tokens at full depth or truncated depth")
L(task=T, kind="training", idea="shared encoder, 5 heads, response-preserving truncation + length header, depth vs token budget (v1 flash silver 4.9k, hard labels, ordinal smoothing 0.6)",
  change="L22@128 vs L11@192 vs L8@256 vs L6@320 tokens",
  dev=d(L22_B128=0.352, L11_B192=0.330, L8_B256=0.346, L6_B320=0.336, v1_reference_test=0.334), gpu_minutes=40, outcome="success",
  takeaway="one-pass 5-head model with a length header ('prompt N response M lines K') at 128 tokens already matches/exceeds v1; depth matters more than extra tokens")
L(task=T, kind="training", idea="seed variance of the baseline (L22@128, v1 silver)", change="seeds 0/1/2",
  dev=d(seed0=0.352, seed1=0.356, seed2=0.346), gpu_minutes=25, outcome="neutral", takeaway="seed noise ~0.01; dev sampling noise larger (~0.03)")
L(task=T, kind="training", idea="one-hot targets instead of ordinal smoothing", change="sigma 0.6 -> 0",
  dev=d(onehot=0.342, baseline=0.352), gpu_minutes=8, outcome="neutral", takeaway="no gain from one-hot; keep sigma 0.6")
L(task=T, kind="data", idea="rule-based degraded variants with known direction (response from another prompt -> help/corr 0; response truncated to 25-45% -> lower)",
  change="+1460 synthetic pairs on top of 4.9k silver",
  dev=d(with_degraded=0.317, baseline=0.352), gpu_minutes=8, outcome="failure",
  takeaway="hurt: real low scores come from subtle factual/instruction errors, not irrelevance or truncation; synthetic shortcuts mislead")
L(task=T, kind="quantisation", idea="int8 dynamic quantisation: which MatMuls", change="all weight MatMuls int8 vs keeping mlp.Wo (and head) fp32",
  dev=d(fp32=0.355, int8_all=0.315, int8_keep_mlpWo_fp32=0.360, agreement_all=0.88, agreement_keep_mlpWo=0.997), outcome="success",
  takeaway="mlp.Wo has activation outliers: quantising it costs 0.04 dev; keeping it fp32 restores fp32 accuracy (agreement 0.997)")
L(task=T, kind="selection", idea="post-hoc stacking: ridge from 5 head scores to each gold attribute (5-fold CV on dev)",
  change="alpha 1-1000", dev=d(raw=0.355, stacked_cv_alpha1=0.335), outcome="failure", takeaway="200 dev examples too few to fit 5x5 stacking weights")
L(task=T, kind="selection", idea="fixed h/c blend: helpfulness and correctness heads both output the mean of their expected scores (gold r=0.95)",
  change="no fitted parameters", dev=d(baseline_s_L22=0.355, blend=0.362, F7_s0=0.364, F7_s0_blend=0.373, F7_s1=0.363, F7_s1_blend=0.376, D_s0=0.366, D_s0_blend=0.377),
  outcome="success", takeaway="+0.01 consistently on every model and both dev halves")
L(task=T, kind="labeling", idea="qwen decimal labels for the whole pool (20.1k) as more/cheaper data", change="qwen on 20.1k pool; flash+qwen merged where both",
  dev=d(student_flash_5k=0.356, student_qwen_5k=0.290, student_merged_fq_5k=0.343, student_merged_15k=0.335),
  cost_usd=0.91, gpu_minutes=60, outcome="failure",
  takeaway="student trained on qwen labels is much worse (0.28-0.29) than on flash (0.35) although labeler-level dev was similar; merging hurts too: labeler agreement with gold does not predict student quality")
L(task=T, kind="training", idea="labeler-specific heads: 5 heads on flash labels + 5 auxiliary heads on qwen labels (shared encoder), deploy flash heads",
  change="7.3k flash + 20.1k qwen", dev=d(dual_s0=0.366, dual_s1=0.347, flash_only_7k_s0=0.364, flash_only_7k_s1=0.363), gpu_minutes=60, outcome="neutral",
  takeaway="qwen as auxiliary task neither helps nor hurts much; less stable than flash-only")
L(task=T, kind="labeling", idea="more flash labels via cheapest OpenRouter provider (provider.sort=price)", change="+2400 flash hard labels (v1 prompt) -> 7.3k flash",
  dev=d(flash_7k_s0=0.364, flash_7k_s1=0.363, flash_5k_seeds=0.351), cost_usd=0.127, gpu_minutes=30, outcome="success",
  takeaway="cheapest route costs 5.3e-5/call vs 1.4e-4 via throughput routing; +2.4k flash labels gave +0.01")
L(task=T, kind="training", idea="epochs and depth on flash 7.3k", change="L22 5 epochs; L18 4 epochs",
  dev=d(L22_e3=0.364, L22_e5_best_ep2=0.356, L18_e4=0.319), gpu_minutes=40, outcome="neutral",
  takeaway="3 epochs best (overfits later); removing 4 layers costs ~0.04 dev, so keep 22 layers at 128 tokens")
print("ok")
