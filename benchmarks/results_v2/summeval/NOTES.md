# SummEval (v2): notes

## Round 3: relaxed setting (8 threads, backbones up to 400M). Not re-scored; final.json is still round 2
No new LLM spend. I reused the round-2 data and labels. Selection is on fold-0 held-out runs over 3 seeds, compared with round-2 16L/192 seeds 0.538 / 0.547 / 0.548.
- **All 22 layers** (no truncation): 192 tokens 0.459 / 0.469, 256 tokens 0.489, 320 tokens 0.467 / 0.500. **Failure.** Full depth is consistently worse on this small set with noisy labels, as in round 1. ModernBERT-large was not tried: at the brief's measured ~100 ms for 128 tokens it cannot fit inputs of 300+ tokens, and more capacity already hurt.
- **Longer inputs with 16 layers:** 256 tokens 0.532 / 0.523 / 0.518, 320 tokens 0.525 / 0.515 / 0.523. **Failure** (about −0.02). More article context does not help.
- **Seed ensembles**, cross-fitted: 5 folds × seeds 0/1/2, int8. Paired bootstrap over the 14 dev articles against the round-2 frozen recipe (0.537):
  - 2 models: 0.545, +0.008, 95% CI [−0.001, 0.017]
  - 3 models: 0.549, +0.012, 95% CI [−0.0001, 0.024], p = 0.027
  - The CIs include 0, so this **does not clear dev noise** and I did **not re-score test**. A +0.01 gain is also far below the 0.084 gap to Jev.
- **8-thread latency:**
  - The round-2 single model has p95 **37-40 ms** (p50 32 ms) on 8 threads, against 51-59 ms on 4 threads.
  - 2 / 3 separate ORT sessions measured p95 90-166 ms and 135-240 ms. The 8-thread pools of the sessions interfere through spinning, and disabling spinning makes things worse.
  - A 2-model ensemble would need both encoders in one ONNX graph to have a chance at about 70 ms.
  - File: `latency_relaxed_8threads_round2model.json`.
- **Conclusion:** on this task the extra CPU budget does not convert into dev accuracy. The limit is the noisy LLM teacher (dev 0.60) and tiny data (86 articles), not encoder capacity.

## Round 2 (current final.json)
- **Test (scored twice; round 2 is the rerun, logged with `--rerun-reason`):** **0.472** mean group-Spearman against Jev **0.556**. Round 1 scored 0.466. **Still does not beat Jev.**
  - Per dim, round 2 / round 1 / Jev:
    - coherence 0.596 / 0.562 / 0.646
    - consistency 0.403 / 0.439 / 0.563
    - fluency 0.426 / 0.402 / 0.494
    - relevance 0.463 / 0.460 / 0.523
  - The dev gain (+0.037) mostly did not transfer to test (+0.006). Consistency, the dimension the new data targeted, *dropped* on test by 0.036.
- **Dev evidence for the rerun** (deployed int8 fold models, 14 never-trained docs): round 2 **0.537** vs round 1 **0.500**. A paired bootstrap over articles (5,000 resamples) gives diff +0.037, 95% CI [0.012, 0.066], p(diff≤0) = 0.002. All 5 fold models improved (0.529-0.552 vs 0.479-0.526). Selection runs (fold 0 held out, 3 seeds) scored 0.544 vs 0.5145, with every seed above every round-1 seed. Details: `round2_dev_paired_bootstrap.json`.
- **What changed.** I added 1,031 "natural" summaries of the 86 eval articles. Three cheap LLMs (qwen3.7-flash, gemini-2.5-flash-lite, deepseek-v4.1-flash) wrote them from 4 plain prompts (3 sentences, CNN-style highlights, 50-word paragraph, quick 2-3 sentences) at T=1.0, without being told to make mistakes. 65% were restyled to PTB-lowercase. qwen labelled them with G1 ×3 votes, and each summary stays in its article's fold.
  They are **used only as targets for the consistency head**; the other heads are masked on these rows. Everything else is identical to round 1.
- **Rejected in round 2:**
  - **LLM sentence-level verification** (`verify.py`). Every summary sentence is judged supported/minor/unsupported/contradicted against the article. On its own it matches gold consistency at 0.59 on dev (the rubric teacher scores 0.57). Blending it into the teacher's consistency target lowered dev (0.51-0.57).
  - Distilling it as an auxiliary head (BCE, w=0.5) gave 0.506 vs 0.5145. The aux head itself reaches only 0.41, so the 192-token student cannot learn the signal. Mixing the aux head into the consistency score at inference did not help either, so it was not used. A request-time LLM call is not allowed, so this signal is unused.
  - Natural summaries for all 4 heads: 0.5146, with consistency up 0.04 but the other dims down.
- **Latency (round-2 fold-0 model, same architecture):** p50 43 ms, **p95 56 ms** (3 runs at load 3-5: 55.6 / 59.1 / 50.9). This passes the 70 ms gate.
- **Spend:** $1.625 of the $1.64 cap in total; round 2 cost $0.39 (verification $0.05, natural summaries generation + labels $0.34).
- **Bookkeeping slip.** A heredoc syntax error meant the round-2 `FROZEN.json` was written before `dev_metrics.json` and `latency.json` were refreshed. Its embedded `dev_metrics` block and the latency copied into `final.json` are therefore the **round-1** values, and the `description` field holds the correct round-2 dev evidence.
  I did not rerun the test to fix this. The corrected files are `dev_metrics_round2.json` and `latency_round2.json`, and `latency.json` is now the round-2 measurement. Round-1 files are kept as `round1_final.json`, `round1_FROZEN.json` and `round1_latency.json`.
- Round-2 models: `data/v2/summeval/models/r2_f{0..4}/`. Probabilities: `data/v2/summeval/final_eval_probs_G/`.

# Round 1 notes

## Result
- **Test (1,376 texts, 86 docs, scored once):** group-Spearman mean over the 4 dims = **0.466**. Jev on the same examples: **0.556**. **Does not beat Jev.**
  - Per dim, ours vs Jev: coherence 0.562 vs 0.646, consistency 0.439 vs 0.563, fluency 0.402 vs 0.494, relevance 0.460 vs 0.523.
  - v1 (context): 0.380, so +0.086 over v1.
- **Dev (204 gold, 14 docs never trained on):** deployed int8 fold models 0.500 mean (folds 0.526/0.496/0.504/0.497/0.479); fp32 0.505; selection runs (fold 0 held out, 3 seeds) 0.5145. Dev overestimated test by ~0.035.
- **CPU latency (frozen pipeline, all 4 heads in one pass, 4 threads, batch 1, dev texts):** p50 49 ms, **p95 55 ms** (3 runs at 1-min load 6-7: p95 55.1 / 55.2 / 57.3). It passes the 70 ms gate. Under heavy load (load 8-70 from other agents) I measured p95 60-410 ms, so re-measure on a quiet machine.
  `benchmarks.v2.latency` cannot load this model because it uses a custom multi-head model with side features. `experiments/summeval/export_mh.py latency` follows the same protocol (dev texts, n=100 after 3 warm-up requests, 4 threads), and its timing includes rendering, tokenisation and feature extraction (about 2 ms in total).
- **Label spend:** $1.24 of the $1.50 scope.
- No test reruns. TEST_ACCESS.log has one line.

## Final recipe (frozen)
1. **Teacher labels.** qwen/qwen3.7-flash, all 4 dims in one call, soft labels (a distribution over 5 levels). The prompt uses a G-Eval-style rubric (`experiments/summeval/guidance.py`, G1): richer criterion definitions, anchors for each level, the dataset conventions ("lower-cased/tokenised text is a display artefact", "extractive copies are consistent"), and "most extracted sentences are fluent". The labels average 5 votes: greedy, 3 samples at T=0.7 (seeds 1-3) and one CoT variant. All 1,376 eval texts were labelled (transductive). No v1 silver labels were used.
2. **Student.** `experiments/summeval/mh.py`. gte-modernbert-base is truncated to its first 16 of 22 layers. The input is `summary: ... source_document: <lead sentence + article sentences ranked by content-word overlap with the summary>`, truncated to 192 tokens. The summary comes first, so truncation cuts only the article. The encoder output uses CLS plus mean pooling, and 17 lexical features computed over the **full** article (`feats.py`: 1-4-gram novelty, novel numbers, repetition, clean ending, lead/article coverage, match positions, case ratio) are added through a small MLP. Four heads with 5 levels each are trained with soft-label cross-entropy for 6 epochs at lr 3e-5 (head 1e-3), batch 16. Batches are built from chunks of 4 summaries of the same article.
3. **Cross-fit.** There are 5 folds by source document (`protocol.eval_folds`). Fold k's model is trained on LLM labels of the other 4 folds and scores only fold k. Dev and seed documents are never in training.
4. **Export.** `fastmb.py` is an export-friendly ModernBERT forward that produces bit-identical outputs: rotary tables are precomputed, rotate_half is a constant matmul, and one window mask is built once. This removes the O(L²) Where/IsNaN mask ops that dominate the stock HF ONNX export on CPU. It is exported to ONNX and dynamically quantised to int8 (per-channel) for every weight MatMul **except the 16 GLU down-projections**, which stay fp32.
   Full int8 cost about 0.03 dev on all 5 folds. Keeping the down-projections in fp32 recovers fp32 accuracy for about 6% more latency.
   The test probabilities come from these mixed-int8 ONNX models, i.e. exactly the measured pipeline.

Models: `data/v2/summeval/models/final_f{0..4}/` (`model.int8.onnx`, `model.onnx`, `mh.pt`). Eval probabilities: `data/v2/summeval/final_eval_probs/`.

## What I tried (dev = mean group-Spearman over 4 dims; details in experiments.jsonl)
**Labeling (LLM labels vs dev gold):**
- Labelers with the G1 rubric: qwen3.7-flash **0.557**, deepseek-v4.1-flash 0.466, claude-haiku-5.5 0.456, gemini-2.5-flash-lite 0.412. qwen was the best and also the cheapest.
- The rubric matters a lot: the plain task prompt scores 0.423, G1 scores 0.557.
- 20 gold seeds as few-shot examples: qwen 0.477 (worse), deepseek 0.498 (better).
- A CoT field on its own: 0.468.
- Self-consistency works: greedy+2 samples 0.580, 5 votes **0.599**. Single sampled runs score only about 0.49.
- Cross-model ensembles: +haiku 0.594 (noise level, and $0.5 for the pool, so not used).

**Student (fold 0 held out, evaluated on dev):**
- Depth × length at 4 epochs: 22L/512 0.437, 22L/160 0.444, 12L/256 0.482, 16L/192 0.500. Long article context does not help this student.
- 6-8 epochs: dev peaks at epochs 5-6.
- Within-article pairwise ranking loss: 0.485 vs 0.504. Lexical features off: 0.485 vs 0.504 (kept).
- lr 5e-5: no gain.
- x3 / x5 teacher labels instead of x1: +0.005 / ±0. A better teacher barely transfers.
- Overlap-aligned article context instead of the lead: +0.006 (consistency and relevance up), kept.
- CLS+mean pooling: +0.01 with lower seed variance, kept.
- Weight soup of 3 seeds: 0.486, worse than every member (0.488-0.523).
- Prediction ensembling of 3 seeds: +0.02, but it triples latency, so it was not used.
- Student-teacher fidelity on held-out fold 0 (group-Spearman between the student and qwen x5) is only 0.55-0.69 per dim. **The student, not the labels, is the bottleneck.**

**More / better training data. The suggested main lever failed here:**
- **LLM-generated summaries with controlled quality profiles.** 18 profiles: minor/major hallucination, misattribution, disfluent, garbled, incoherent order, redundant, truncated, off-focus, verbose, extractive variants, etc. The summaries were styled like the dataset (PTB tokens, lower case). There were 9 per article call, and qwen labelled them with G1. Adding 650 of them to 1,087 real texts gave **0.445 vs 0.487** (2 seeds). Consistency dropped the most (0.25 vs 0.33).
  Weighting the synthetic data at 0.3, masking the consistency head, or dropping the hallucination profiles all stayed below real-only (0.448-0.469). The likely cause is that the synthetic flaws are much grosser than the real differences between systems. The student learns novelty/flaw cues and loses the fine within-article ranking that the metric rewards.
- **Programmatic extractive summaries in SummEval-system style** (lead-3, lead-2, random/late sentences, duplicated sentence, length-capped lead; 560, LLM-labelled): 0.460/0.487 vs 0.509. Failure.
- 281 LLM-written CNN/DM-style articles were generated ($0.04) but not used, because synthetic summaries had already failed on real articles.
- **Dev gold in training.** I trained on half of the dev docs and evaluated on the other half. This lowered dev by 0.06-0.08 (by 0.04-0.06 with gold weight 3). Sharp gold targets do not mix with LLM soft labels here.
- LLM labels of the dev-half texts (+100 real-distribution texts): neutral (+0.01 / −0.01). Data volume is not the limit.

**Latency engineering:**
- The stock HF→ONNX export of ModernBERT spends most of its time in mask ops. At 22L/256 tokens p95 was 142 ms, and at 512 tokens 600 ms.
- Eager attention plus the custom forward (`fastmb.py`) fixed this. The budget at the gate is about 16 layers × 192 tokens. Full inputs (560-790 tokens) are 10-20x over the gate.

## Notes for merging into the library
- `experiments/summeval/mh.py`: a shared encoder with K Score heads, optional side-feature MLP, soft-label KL, per-example/per-dim loss weights, and article-grouped batches. This is a candidate for `ownyourjev.models.multihead`.
- `experiments/summeval/fastmb.py`: a fast, export-friendly ModernBERT forward (bit-identical). It should replace the stock export for gte-modernbert in `ownyourjev.export`, since the sdpa export is 1.5-3x slower on CPU at ≥256 tokens.
- `export_mh.quantize`: mixed int8 that excludes the GLU down-projections. Worth testing on other gte-modernbert tasks that lost accuracy with full int8.

## What I would try next
- **Prediction ensemble inside the latency budget.** Use 2-3 smaller students (for example 8 layers × 160 tokens each). Seed ensembles gave +0.02 on dev, but the budget would have to be split between them.
- **A stronger teacher only where the student is weakest (consistency).** For example, a per-sentence entailment check of summary sentences against the article by an LLM, as an extra soft target. Rough cost with qwen: about $0.15 for the pool.
- **Distillation at scale from natural, unprompted summaries.** Generate summaries of the 86 + 281 articles by many weak models or decoding settings, *not* prompted to be flawed, so they look like real system outputs (16 per article ≈ 5.9k texts), and label them with qwen x5. Estimated cost is about $3 ($2.9 for labels, $0.1 for generation). Given that both synthetic attempts failed, I rate this low-confidence. I would pilot it on 50 articles (≈ $0.35) before scaling.
