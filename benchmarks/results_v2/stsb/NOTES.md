# STS-B (v2): notes

**Current result (round 2): ours 0.8935 vs Jev 0.8900 on test (beats Jev), CPU p95 34.5 ms, label spend $1.539.** Round 1 is below.

## Round 1 result (test, scored once)
- **Ours 0.879 Spearman (95% CI 0.865–0.892) vs Jev 0.890 on the same 1,379 test pairs: does not beat Jev.**
  Pearson 0.886, MAE 0.54 levels.
- CPU latency (ONNX selective int8, 4 threads, batch 1, loaded shared machine): p50 22.9 ms, **p95 39.0 ms** (gate is 70 ms, so it passes).
- Label spend (scope `stsb`): **$0.778** of the $1.00 cap. About $0.27 of that was dev experiments; the pool labels cost $0.52.
- One test run, no rerun.

## Final recipe (frozen)
1. **Labels.** All 5,529 pool pairs were labeled by 4 LLMs. Each LLM gives a *real-valued* 0–5 score. The calls are
   batched, 10 pairs per call (`experiments/stsb/batch_label.py`), and use SemEval-STS guidance (`label_dev.py: GUIDE["semeval_soft"]`)
   with no seeds. The labelers are gemini-3.1-flash-lite, claude-haiku-5.5, qwen3.7-flash and deepseek-v4.1-flash.
   Each score becomes a two-hot distribution over the 6 levels (this keeps its expectation), and the 4 distributions are
   averaged into the soft target.
2. **Student.** gte-modernbert-base is fully fine-tuned with soft-target cross-entropy. It uses **sentence-order swap
   augmentation** (each pair is also added with sentence_1 and sentence_2 swapped, giving 10.5k training rows), lr 3e-5,
   bs 32, max_len 128 and 4 epochs. The epoch is chosen by Spearman on a 300-pair held-out *silver* split (epoch 3),
   never on dev. This is a custom loop in `experiments/stsb/train2.py`, because the library loop early-stops on hard-label
   CE val loss. Seed 0 was chosen among 4 seeds on dev.
3. **Deployment.** ONNX with **selective dynamic int8**: every MatMul is quantised except the MLP output projections
   (`mlp/Wo`) and the head (`quant_variants.py`). Model: `data/v2/stsb/models/final/model.int8.onnx`.

## Dev experiments (200 gold dev pairs; A/B = even/odd halves)
### Labelers (Spearman of the labeler's expected level vs dev gold)
| labeler / prompt | dev | cost per 200 |
|---|---|---|
| deepseek-v4.1-flash, hard int (v1 style) | 0.844 | – |
| v1 silver / silver_seeded (dev uids, n=180) | 0.856 / 0.862 | free |
| deepseek, distribution, no guidance | 0.838 | $0.019 |
| deepseek, distribution, SemEval guidance | 0.865 | $0.015 |
| same + 20 seeds | 0.845 | $0.013 |
| qwen3.7-flash, dist + guidance | 0.846 | $0.005 |
| gemini-2.5-flash-lite, dist + guidance | 0.825 | $0.018 |
| deepseek-v4-pro, dist + guidance | 0.736 (10 failures) | $0.014 |
| gpt-5.6-luna, dist + guidance | 0.883 | $0.038 |
| claude-haiku-5.5, dist + guidance | 0.885 | $0.026 |
| gemini-3.1-flash-lite, dist + guidance | 0.904 | $0.052 |
| **batched (10/call) real-valued score**: gemini-3.1-flash-lite | **0.909** | **$0.0085** |
| batched score, gemini-3.1, 20/call | 0.905 | $0.007 |
| batched score, gemini-3.1 + 20 seeds | 0.900 | $0.012 |
| batched distribution, gemini-3.1 | 0.898 | $0.014 |
| batched score: deepseek / luna / qwen / haiku | 0.880 / 0.877 / 0.870 / 0.859 | $0.006 / 0.007 / 0.001 / 0.004 |
| mean of batched gemini+haiku+qwen | 0.917 | |
| **mean of batched gemini+haiku+qwen+deepseek** | **0.919** | |
| mean of all 5 batched | 0.919 | |

Takeaways:
- Batching 10 pairs per call and asking for a real-valued score (one decimal) was both **cheaper (6x) and better** than
  per-pair 6-way distributions.
- The SemEval guidance helped deepseek (+0.027). Seeds did not help either way.
- Averaging labelers adds +0.01.

### Student (dev Spearman of the expected level; fp32 torch)
| run | dev |
|---|---|
| v1 silver_seeded hard labels, library `train` | 0.866 |
| v1 silver + silver_seeded averaged soft | 0.869 |
| gemini-only soft | 0.887 |
| 3-labeler soft | 0.886 |
| **3-labeler soft + swap augmentation** | **0.909** |
| + 6 epochs / + expected-score MSE term / lr 5e-5 | 0.908 / 0.907 / 0.907 |
| + 20 seeds as gold two-hot targets | 0.898 |
| gemini-only + swap | 0.900 |
| 4-labeler + swap, seeds 0/1/2/3 | **0.908** / 0.895 / 0.898 / 0.895 (mean 0.899) |
| 4 labelers + v1 hard labels (weight 0.5) + swap | 0.893 |
| swap test-time averaging (both orders) | +0.000–0.003 (not used: doubles latency) |
| prob ensemble of 3 seeds | 0.902 (no better than the best single; 3x latency) |
| uniform weight soup of 3 seeds | 0.897 |

- The biggest student lever was **swap augmentation (+0.02)**. Label ensembling paid off only together with swap
  (0.900 → 0.909).
- Seed variance (±0.007) is as large as most other effects, so dev cannot separate the variants beyond the swap effect.

### Quantisation (seed 0 model, dev)
| runtime | dev | p95 (loaded machine) |
|---|---|---|
| ONNX fp32 | 0.908 | 61–63 ms (too close to the gate) |
| ONNX int8, library `quantize_int8` (all MatMul/Gemm) | 0.901 | 36–43 ms |
| ONNX int8, MLP `Wo` + head kept fp32 | **0.910** | 39–41 ms |

Keeping the attention act×act MatMuls in fp32 changed nothing. The full-int8 loss comes from the MLP output projections.

## Why test < dev
- Dev 0.910 → test 0.879. Part of the gap is selection bias: seed 0 was the best of 4, and the seed mean is 0.899.
- The rest is a dev/test distribution shift. Dev is drawn from the STS-B *train* split, and the test split is known to
  be harder. The v1 student scored 0.882 on test.
- The labels reach 0.919 on dev, but the student loses about 0.01 on dev and more on test. With ~5.5k pairs the student,
  not the labels, is the bottleneck.

## Merge candidates (implemented in experiments/stsb/, library untouched)
- `batch_label.py`: N items per call with a compact output (a real-valued score or a list of probabilities). It gives
  about 6x lower label cost on short-input tasks and better agreement here.
- Swap augmentation for symmetric pair tasks. This could be a `TrainConfig` option or a data helper.
- `train2.py`: selects the epoch by Spearman on silver val (instead of hard-label CE), with optional MSE and ordinal blur.
  Its OOM retry must free the CUDA cache *outside* the `except` block, because the traceback keeps the model alive;
  otherwise two retrying jobs deadlock on one GPU.
- `quant_variants.py`: an `exclude` option for `quantize_int8`. Excluding the `mlp/Wo` MatMuls removed the int8 loss on
  gte-modernbert at small latency cost.

## What I would try next
- More pairs rather than better labels:
  - transductive LLM labels on unlabeled `eval_inputs` (allowed by the protocol, but not used here);
  - paraphrase or edit-based synthetic pairs.
- Distil a teacher ensemble of several students, or a soft-target blend that also uses the student's own predictions.
- Average over more seeds, or use checkpoint averaging within one run, to cut the ±0.007 seed variance.
- Regression (MSE on expected score) as the main loss, with a larger `max_len`. Inputs are short.


## Round 2 (data-side: more training pairs + weight soup)
Every experiment below is in `experiments.jsonl` (round=2). The round-1 experiments were backfilled as round=1.

### Test result (re-scored once, with `--rerun-reason`)
- **Ours 0.8935 Spearman (95% CI 0.881–0.905) vs Jev 0.8900 on the same 1,379 test pairs: beats Jev.** Round 1 was 0.8791.
  The CI still overlaps Jev, so the margin (+0.0035) is not statistically clear.
- **CPU latency:** ONNX selective int8, 4 threads, quiet machine: p50 19.8 ms, **p95 34.5 ms**.
- **Spend:** scope total **$1.539** of $1.58. Round 2 spent $0.76:
  - generation $0.06;
  - labeling 7.6k new pairs with 4 labelers, about $0.093 per 1k pairs;
  - the rest on pilots.
- **Archive files:** round-1 records are in `round1_final.json` and `round1_FROZEN.json` (I copied them by hand before the rerun).
  final_eval's auto-archiver also wrote `round2_final.json` and `round2_FROZEN.json`, which are *also the round-1 result*:
  the archiver counted my manual copy. Nothing was overwritten. The current round-2 result is `final.json`.
  Round-2 eval probabilities are in `round2/`.

### First, the honest baseline
I re-ran the round-1 recipe with 6 seeds: 0.908 / 0.895 / 0.897 / 0.895 / 0.893 / 0.893.
The **mean is 0.8968 (std 0.006)**. The round-1 deployed model (0.910) was a lucky draw.

### New training pairs (all scored by the same 4-labeler average)
1. **LLM-generated partners.** deepseek-flash wrote, for 800 pool sentences, one partner per target level 1–5 in the
   source's style (3.7k pairs, `gen_pairs.py`, $0.06). I used 1k of them; the target level is ignored and the
   4 labelers re-score every pair. Labels skew high (45% ≥ 4). Adding them alone is **neutral**: 4 seeds,
   mean 0.896 vs 0.897.
2. **Re-pairings of pool sentences** (`repair_pairs.py`).
   - 10.1k unique pool sentences are embedded with base gte-modernbert, and each is paired with 2 random neighbours
     from its top 30, never its original partner (19k candidates).
   - The round-1 student scores the candidates, and I sample stratified by its predicted level.
   - The candidates are mostly low-similarity: predicted levels 0–5 come out 8.6k / 7.0k / 2.7k / 0.8k / 0.2k.
   - First batch 3.4k (all high bins plus 800 per low bin), then 3.15k more from bins 0–2.
   - Labeler agreement is lower on these pairs: each labeler vs the mean of the others is 0.78–0.94.

| training data (all + swap aug, 4 epochs) | dev, per seed | seed mean |
|---|---|---|
| round-1 pool only (5.2k) | .908/.895/.897/.895/.893/.893 | 0.897 |
| + 1k generated | .901/.897/.891/.895 | 0.896 |
| + 3.4k re-paired | .907/.902/.903 | 0.904 |
| + 3.4k re-paired + 1k generated | .906/.906/.903 | 0.905 |
| **+ 6.6k re-paired + 1k generated** | .908/.914/.906 | **0.909** |
| same, fixed 4 epochs, no held-out val (`--full`) | .910/.904/.905/.907 | 0.907 |
| same, 6 epochs | .906/.902 | no gain |

More *in-distribution* pairs, meaning real pool sentences re-paired, are what helped: about +0.012 on the seed mean.
Generated pairs added little.

### Seed variance: weight soup
- Soups of runs with **independent seeds lose**:
  - 3 early-stopped runs: soup 0.906 vs member mean 0.909;
  - 4 fixed-epoch runs: soup 0.902 vs member mean 0.907.
  Each seed initialises the classification head differently, so the averaged weights do not line up.
- **Sharing the init** fixes it. With init seed 0 for every member and data-order seeds 0/1/2, the members score
  .910/.911/.910 and the **soup scores 0.911**: one model, same latency.
  This is `train2.py --init_seed`.
- Quantisation: selective int8 0.9121, full int8 0.9121, fp32 0.9111. I kept the selective variant because it is closer to fp32.

### Dev comparison that justified the re-score
Paired bootstrap on dev (2,000 resamples), deployed round-2 model (0.9121):
- vs the **round-1 recipe (mean over its 6 seeds)**: **+0.0153, 95% CI [+0.0038, +0.0281]**, P(>0) = 0.996.
  This is above dev noise, so I re-scored.
- vs the round-1 *deployed* model (best of 4 seeds, 0.910): +0.002, CI [−0.010, +0.013], which is not significant.
  The policy compares recipes (mean over seeds, not best-of-seeds), and this comparison is reported for transparency.

### Round-2 frozen recipe
1. Labels: round-1 labels plus 7,571 new pairs: 6,571 re-paired pool sentences and 1,000 generated pairs.
   All are scored by the 4-labeler average and turned into two-hot soft targets.
2. Training: gte-modernbert full FT, swap augmentation (about 25.6k rows), lr 3e-5, bs 32, max_len 128, fixed 4 epochs.
   Three runs share the init, and their weights are averaged uniformly.
3. Deployment: ONNX selective int8 in `data/v2/stsb/models/final_r2/`.

### Test vs dev
- Dev 0.912 → test 0.894. Round 1 went 0.910 → 0.879.
- The test gain (+0.014) matches the dev gain over the round-1 *recipe mean* (+0.015). This suggests round-1's lucky
  seed was luck on dev only.

### Next (not done)
- More re-paired pairs. Only $0.04 of the cap is left, but the gain had not saturated (+3.15k pairs still gave +0.004).
- Cheaper labeling for bulk pairs, e.g. dropping haiku and qwen, which agree least on re-paired data.
- A soup of more shared-init members.


## Round 3 (more re-paired pool pairs; no test re-score)
Ledger entries are under round=3.

### What was done
- **Candidates** (`repair3.py`):
  - every top-12 gte-modernbert neighbour of every unique pool sentence, excluding original pool pairs, dev/seed
    sentences and the round-2 pairs: 81.7k candidates;
  - scored by the round-2 student, giving 33.9k / 31.6k / 10.4k / 4.1k / 1.7k across predicted levels 0–5.
- **Labeling:** stratified quotas of 2000/2000/3000/3000/1695 = **11,695 new pairs**, labeled with the same 4-labeler
  average (batched 10/call), for **$1.034** (about $0.088 per 1k pairs).
  - The resulting labels are spread evenly over levels: 2274/2544/2505/2448/1924.
  - Labeler agreement (each vs the mean of the others) is 0.92–0.96.
- **Training:** pool + round-2 new pairs (7.6k) + round-3 new pairs (11.7k) with swap augmentation, 49.6k rows.
  Fixed 4 epochs, 3 shared-init runs (data-order seeds 0/1/2), uniform weight soup.

### Dev
| model | dev Spearman |
|---|---|
| round-2 frozen (deployed ONNX) | 0.9121 |
| round-2 members (mean) | 0.9105 |
| round-3 members | 0.9208 / 0.9157 / 0.9153 (mean 0.9173) |
| round-3 soup, fp32 / ONNX int8 | 0.9169 / **0.9177** |

Paired bootstrap on dev (2,000 resamples):
- round-3 soup vs the **round-2 deployed model: +0.0056, 95% CI [−0.0035, +0.0159]**, P(>0) = 0.88;
- vs the round-2 member mean: +0.0072, CI [−0.0011, +0.0171].

**Decision:** the CI includes 0, so the gain does not clear dev noise. Under the round-3 rule I did **not** take a
third test look. **The round-2 result (test 0.8935 vs Jev 0.8900) stands.**

The round-3 model is kept in `data/v2/stsb/models/soup_r3/` (ONNX int8, CPU p95 38.0 ms, p50 21.9 ms).

### Takeaways
- Re-paired in-distribution pairs keep helping, but with diminishing returns per pair:
  - 5.2k → 12.8k pairs: +0.012 on the seed mean;
  - 12.8k → 24.5k pairs: +0.007.
- A 200-pair dev set cannot resolve gains much below about 0.01. A clear test win would need either a bigger
  jump (a different lever) or a larger dev set.
- **Spend:** scope total **$2.573** of $2.74; round 3 spent $1.034.


## Round 4: relaxed setting (8 threads, backbones up to ~400M; no test re-score)
Ledger entries are under round=4, setting="relaxed". No new LLM spend: scope total still $2.573.

- **Dev token lengths** (rendered pair, ModernBERT tokenizer): p50 34, p90 65, p95 75, max 114.
- **ModernBERT-large** (395M), trained on the full round-3 data (49.6k rows with swap, 4-labeler soft targets):
  1 run, init 0, lr 2e-5, bs 16, 3 epochs.
  - Dev 0.9163 (int8 0.9161). That is *no better* than gte-modernbert (round-3 members 0.9157–0.9208, soup 0.9177).
  - Truncating to 64 tokens to save latency drops dev to 0.886.
  - Latency at 8 threads, int8, loaded machine: p50 57 ms, **p95 94–129 ms**, so it **fails the gate**.
    I did not train the 3-run soup, given no accuracy gain.
- **Ensembles** (8 threads allow about 2 base models):
  - soup_r3 + a gte run with init seed 1: dev 0.916. That is no gain over the soup alone (0.9177).
  - soup_r3 + ModernBERT-large: **dev 0.9241**, +0.012 vs the round-2 model, CI [+0.0005, +0.025]. This is the only
    candidate whose CI excludes 0. But it was picked after the fact from several candidates, and its latency is
    p50 75 ms / p95 133 ms, so it is **not deployable**.
- **Best deployable relaxed model:** the round-3 soup at 8 threads: p50 19.3 ms, p95 31.3 ms, dev 0.9177.
  Against the round-2 deployed model: +0.0056, CI [−0.0035, +0.0159], which includes 0.

**Decision:** no relaxed re-score. The result stays round 2: test 0.8935 vs Jev 0.8900.

**Takeaway:** for STS-B the extra latency budget does not buy accuracy. The large backbone does not beat the base one
trained on the same data. The only dev gain, from mixing backbones, does not fit 70 ms.
