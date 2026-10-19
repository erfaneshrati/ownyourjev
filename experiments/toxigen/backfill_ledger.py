from benchmarks.v2.ledger import log_experiment as L
T = "toxigen"
B = {"labeler_baseline": "flash_plain acc_best 0.885 / spearman 0.719 / auroc 0.913", "n": 200}
rows = [
 ("labeling", "deepseek-v4.1-flash plain soft (P(toxic)+5-level dist, one call)", "", {"acc_es_best": .885, "spearman": .719, "auroc_es": .913}, .019, "neutral", "baseline labeler; E[toxicity] better than P(toxic) for the binary"),
 ("labeling", "flash + 20 gold seeds as soft seed answers", "", {"acc_es_best": .91, "spearman": .766, "auroc_es": .951}, .012, "success", "seeds help ranking"),
 ("labeling", "flash + guidance v1 (implicit/coded hate vs mention)", "", {"acc_es_best": .905, "spearman": .754, "auroc_es": .957}, .012, "success", "best AUROC single flash prompt; chosen for pool"),
 ("labeling", "flash + seeds + guidance v1", "", {"acc_es_best": .895, "spearman": .769, "auroc_es": .930}, .016, "neutral", "combining seeds and guidance did not stack"),
 ("labeling", "flash hard labels, seeds + guidance", "", {"acc_es_best": .89, "spearman": .718, "auroc_es": .881}, .010, "failure", "hard labels lose ranking information vs soft distribution"),
 ("labeling", "deepseek-v4-pro + seeds + guidance", "", {"acc_es_best": .905, "spearman": .795, "auroc_es": .946}, .022, "neutral", "best single Spearman, too costly for full pool in round 1"),
 ("labeling", "claude-haiku-5.5 + seeds + guidance", "", {"acc_es_best": .92, "spearman": .764, "auroc_es": .955}, .062, "neutral", "best single accuracy, too costly in round 1"),
 ("labeling", "gemini-2.5-flash-lite + seeds + guidance", "", {"acc_es_best": .885, "spearman": .708, "auroc_es": .931}, .043, "failure", "no better than flash"),
 ("labeling", "qwen3.7-flash + seeds + guidance", "", {"acc_es_best": .875, "spearman": .656, "auroc_es": .811}, .007, "failure", "weak labeler"),
 ("labeling", "gpt-5.6-luna / gemini-3.1-flash-lite", "", {}, 0.0, "failure", "all calls failed: client max_price provider cap 0.2/M"),
 ("labeling", "labeler ensembles on dev (avg soft dists)", "", {"flash_g1+haiku": {"acc_es_best": .93, "spearman": .787, "auroc_es": .973}, "4way_spearman": .80}, 0.0, "success", "ensembles +0.02-0.03 acc, +0.03-0.05 Spearman; unaffordable under $1 cap in round 1"),
 ("training", "student on v1 silver seeded hard labels (4.9k)", "gte-modernbert multihead 3ep", {"spearman": .757, "auroc_es": .950, "acc_es_cv": .85}, 0.0, "neutral", "baseline student"),
 ("training", "student on v1 silver plain+seeded", "", {"spearman": .742, "auroc_es": .943, "acc_es_cv": .835}, 0.0, "failure", "averaging two hard flash labelings did not help"),
 ("data", "student on full-pool flash_g1 soft labels (8.7k)", "", {"spearman": .773, "auroc_es": .958, "acc_es_cv": .915, "baseline_auroc_es": .950}, .52, "success", "full pool soft labels beat v1 silver"),
 ("data", "mix v1 silver at weight 0.5 into soft targets", "", {"spearman": .770, "auroc_es": .960, "acc_es_cv": .91}, 0.0, "neutral", "within seed noise"),
 ("training", "lr 2e-5 / 2 epochs variants", "", {"lr2e-5_auroc": .958, "2ep_auroc": .959}, 0.0, "neutral", "within seed noise"),
 ("selection", "toxic from E[toxicity] vs toxic head", "", {"acc_cv_es_mean": .896, "acc_cv_p_mean": .877}, 0.0, "success", "deriving toxic from toxicity distribution is better"),
 ("training", "uniform weight soup of 3 seeds", "", {"spearman": .770, "auroc_es": .958, "acc_at_0.30": .89}, 0.0, "neutral", "soup ~ mean of seeds, more stable"),
 ("quantisation", "int8 dynamic per-channel ONNX", "", {"auroc_es": .940, "baseline_auroc_es": .960, "spearman": .729, "baseline_spearman": .769}, 0.0, "failure", "int8 hurts gte here; fp32 ONNX already p95 ~37 ms"),
]
for k, idea, ch, dev, c, o, t in rows:
    L(task=T, round=1, kind=k, idea=idea, change=ch, dev={**dev, "n": 200}, cost_usd=c, outcome=o, takeaway=t)
print(len(rows))
