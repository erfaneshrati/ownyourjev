import sys; sys.path.insert(0, ".")
from benchmarks.v2.ledger import log_experiment as L
t = dict(task="stsb", round=1)
B = 0.844
for idea, d, c, o, tk in [
 ("deepseek-v4.1-flash hard integer labels (v1 style)", 0.844, 0.001, "neutral", "baseline labeler agreement"),
 ("deepseek per-pair 6-way distribution, no guidance", 0.838, 0.019, "failure", "soft output alone does not help"),
 ("deepseek distribution + SemEval STS guidance", 0.865, 0.015, "success", "guidance +0.027 over plain distribution"),
 ("deepseek distribution + guidance + 20 gold seeds in prompt", 0.845, 0.013, "failure", "seeds hurt (-0.02)"),
 ("qwen3.7-flash dist + guidance", 0.846, 0.005, "neutral", "cheap, similar to deepseek"),
 ("gemini-2.5-flash-lite dist + guidance", 0.825, 0.018, "failure", "weaker labeler"),
 ("deepseek-v4-pro dist + guidance", 0.736, 0.014, "failure", "10 parse failures, poor agreement"),
 ("gpt-5.6-luna dist + guidance", 0.883, 0.038, "success", "good but pricey"),
 ("claude-haiku-5.5 dist + guidance", 0.885, 0.026, "success", "good labeler"),
 ("gemini-3.1-flash-lite dist + guidance", 0.904, 0.052, "success", "best per-pair labeler"),
 ("gpt-5-mini dist + guidance", None, 0.0, "failure", "no calls succeeded (routing/price cap)"),
 ("batched 10 pairs/call, real-valued 0-5 score, gemini-3.1", 0.909, 0.0085, "success", "6x cheaper and better than per-pair distribution"),
 ("batched 20 pairs/call score, gemini-3.1", 0.905, 0.007, "neutral", "slightly worse than 10/call"),
 ("batched score + 20 seeds, gemini-3.1", 0.900, 0.012, "failure", "seeds do not help"),
 ("batched 10/call distribution output, gemini-3.1", 0.898, 0.014, "failure", "score output better than distribution"),
 ("batched score: deepseek / luna / qwen / haiku", 0.880, 0.017, "neutral", "0.880/0.877/0.870/0.859 single labelers"),
 ("average of gemini+haiku+qwen batched scores", 0.917, 0.0, "success", "labeler ensemble +0.008"),
 ("average of gemini+haiku+qwen+deepseek batched scores", 0.919, 0.0, "success", "chosen label recipe"),
]:
    L(**t, kind="labeling", idea=idea, change="dev labeling only", dev={"labeler_spearman": d, "baseline_labeler_spearman": B, "n": 200}, cost_usd=c, outcome=o, takeaway=tk)
S = 0.866
for idea, d, g, o, tk in [
 ("v1 silver_seeded hard labels, library train (baseline)", 0.866, 4, "neutral", "student baseline"),
 ("v1 silver+silver_seeded averaged soft targets", 0.869, 4, "neutral", "within noise"),
 ("gemini-only soft targets (pool labeled, $0.23)", 0.887, 2, "success", "better labels -> +0.02"),
 ("3-labeler (gemini+haiku+qwen) soft targets", 0.886, 2, "success", "same as gemini-only without augmentation"),
 ("3-labeler soft + sentence-order swap augmentation", 0.909, 3, "success", "biggest student lever, +0.023"),
 ("+ 6 epochs", 0.908, 3, "neutral", "no gain"),
 ("+ expected-score MSE term", 0.907, 3, "neutral", "no gain"),
 ("+ lr 5e-5", 0.907, 3, "neutral", "no gain"),
 ("+ 20 seeds as gold two-hot targets", 0.898, 3, "failure", "hurt slightly"),
 ("gemini-only + swap", 0.900, 3, "neutral", "ensemble labels help with swap (+0.009)"),
 ("4-labeler + v1 hard labels (w 0.5) + swap", 0.893, 3, "failure", "v1 hard labels add noise"),
 ("4-labeler + swap, seeds 0-3", 0.899, 12, "neutral", "seed values 0.908/0.895/0.898/0.895; seed std ~0.006"),
 ("4-labeler + swap + 20 seeds", 0.896, 3, "failure", "no gain"),
 ("swap test-time averaging", 0.911, 0, "neutral", "+0.000-0.003; doubles latency, not used"),
 ("probability ensemble of 3 seeds", 0.902, 0, "neutral", "3x latency, no better than best seed"),
 ("uniform weight soup of 3 seeds (0,1,2)", 0.897, 0, "neutral", "soup ~ seed mean"),
]:
    L(**t, kind="training" if "seed" not in idea or "soup" in idea else "selection", idea=idea, change="student training on pool labels", dev={"spearman": d, "baseline_spearman": S, "n": 200}, gpu_minutes=g, outcome=o, takeaway=tk)
L(**t, kind="quantisation", idea="library quantize_int8 (all MatMul/Gemm) on gte-modernbert", dev={"spearman": 0.901, "baseline_spearman": 0.908, "n": 200}, outcome="failure", takeaway="-0.007 vs fp32 ONNX")
L(**t, kind="quantisation", idea="int8 excluding attention act x act MatMuls + head", dev={"spearman": 0.901, "baseline_spearman": 0.908, "n": 200}, outcome="failure", takeaway="no change; loss is elsewhere")
L(**t, kind="quantisation", idea="int8 excluding mlp/Wo MatMuls + attention act MatMuls + head", dev={"spearman": 0.910, "baseline_spearman": 0.908, "n": 200}, outcome="success", takeaway="recovers fp32 accuracy; p95 ~40 ms")
L(**t, kind="latency", idea="ONNX fp32 latency", dev={"cpu_p95_ms": 62}, outcome="failure", takeaway="too close to 70 ms gate on loaded machine")
L(**t, kind="selection", idea="round-1 frozen: 4-labeler soft + swap, seed 0 (best of 4), selective int8", dev={"spearman": 0.910, "seed_mean": 0.899}, cost_usd=0.778, outcome="neutral", takeaway="dev selection optimistic; seed 0 picked best-of-4")
