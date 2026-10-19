# BoolQ (v2): notes

> **Current result (round 2; neither round 3 nor the relaxed setting improved on dev, so test was not re-scored):** accuracy **0.855** (95% CI 0.843–0.867) vs Jev **0.913**.
> **Still does not beat Jev.** Round 1 scored 0.844. CPU p95 64.3 ms (p50 41.2 ms). Total label spend $1.17.
> Round 1 files are archived as `round1_final.json`, `round1_FROZEN.json` and `round1_latency.json`. The ledger is
> `experiments.jsonl` (55 lines). Rounds 1, 2 and 3 and the relaxed setting are described in order below.

## Round 1

**Round-1 result (test):** accuracy **0.844** (95% CI 0.832–0.857) vs Jev **0.913** on the same 3,270 examples. **Does not beat Jev.**
**CPU latency:** p50 42.7 ms, **p95 69.1 ms** (ONNX Runtime int8, 4 threads, `benchmarks.v2.latency --onnx`, measured while the machine was
heavily loaded, load average 11–28 on 20 threads; three repeats gave p95 67.2–69.1 ms). This passes the 70 ms gate, but only just.
**Label spend:** $0.54 (scope `boolq`, OpenRouter). No test re-run.

## Final recipe (frozen on dev)
- Backbone: **UltraBERT-150M** (24 layers), full fine-tune, lr 1e-5, batch 16, bf16, early stopping on 400 held-out *silver* pool examples.
  Uses `max_len` 224 with **left truncation**: the rendered state is `passage … question`, so right truncation would cut the question.
  On dev, left truncation at 224 scored the same as no truncation (0.855 vs 0.85).
- Model: a **uniform weight soup** (an average of the weights) of the 6 UltraBERT fine-tunes listed below, all from the same pretrained
  init: r2_ub_pool_gen, r4_ub_comb, r4_ub_qwen, r4_ub_comb_gen, r5_ub_qwen_s1, r6_cur. The soup costs nothing at inference.
- Labels:
  - qwen/qwen3.7-flash on the whole pool (9,207 texts, $0.10).
  - Some runs used the "comb" labels instead: v1 deepseek-v4.1-flash *seeded* labels (from v1, with different seed examples drawn from
    the training pool), plus gemini-3.1-flash-lite tie-breaks on the 714 texts where qwen and v1 disagreed or qwen was uncertain ($0.05).
  - Some runs also trained on 23.7k LLM-generated yes/no questions: qwen wrote 8 questions per passage (4 yes and 4 no, BoolQ-style
    search queries, with hard negatives) for 3,000 pool passages ($0.17).
- Serving (`experiments/boolq/fast_export.py`, `quant.py`):
  - A **lean ONNX re-implementation** of the encoder: the same weights and the same math (max diff vs HF < 1e-6), with additive masks
    computed once, RoPE tables precomputed, and fused weights split. The stock HF export spent more than half its CPU time in mask
    plumbing (Where/IsNaN/Slice per layer).
  - **SmoothQuant** (α=0.8) on the MLP down-projection. Its scale is folded into `up_proj`, which is exact because the product is linear
    in `up_proj`.
  - Dynamic **int8** on the weight MatMuls only (attention and RoPE matmuls stay fp32).
- Threshold 0.5. A threshold tuned on dev half A did not transfer to half B (and the reverse).
- Dev (200): int8 ONNX accuracy **0.85**, AUROC 0.924 (fp32 ONNX 0.84; torch 0.845).

## What I tried (dev accuracy / AUROC, n=200; ±~2.5 pts is noise)
**Labelers** (agreement with dev gold; plain prompt unless noted):

| labeler | dev acc |
|---|---|
| deepseek-v4.1-flash | 0.865 |
| deepseek-v4.1-flash + 20 seeds | 0.93 |
| qwen3.7-flash | 0.925 (soft-label variant 0.925; with seeds 0.92, 10x the cost) |
| gemini-3.1-flash-lite | 0.93 |
| claude-haiku-5.5 | 0.865 |
| deepseek-v4-pro | 0.895 (and expensive) |
| majority of flash+seeds, qwen and gemini | 0.95 |

Labels are not the bottleneck: the student sits ~8–10 pts below the labelers.

**Generated questions:** the generator's answers agreed with gemini on 99% of 200 checked (qwen generator) and 96.5% (deepseek
generator; deepseek also cost 3.5x more).

**Students:**

| run | backbone | data | dev acc / AUROC |
|---|---|---|---|
| b0 | gte-modernbert 22L | v1 seeded labels, 4.5k | 0.74 / 0.80 (mean pooling 0.69) |
| r1_pool | gte | qwen pool labels, 8.8k | 0.835 / 0.885 |
| r1_pool_gen | gte | + 23.7k generated | 0.775 / 0.865 (**generated data hurt gte**) |
| r3_comb | gte | comb labels, 8.8k | 0.77 / 0.878 |
| r3_genonly | gte | generated only | 0.595 / 0.74 (the generated questions alone do not transfer to real BoolQ) |
| r2_ub_pool_gen | **UltraBERT** | qwen pool + generated | 0.845 / 0.914 |
| r4_ub_qwen | UltraBERT, 224 left | qwen pool | 0.845 / 0.935 |
| r5_ub_qwen_s1 | same, seed 1 | qwen pool | 0.82 / 0.899 (**large seed variance**) |
| r4_ub_comb | UltraBERT | comb labels | 0.83 / 0.909 |
| r4_ub_comb_gen | UltraBERT | comb + generated | 0.85 / 0.917 |
| r6_cur | UltraBERT | init from r4_ub_comb_gen, then pool only, lr 5e-6 | 0.85 / 0.914 |
| r5_ub_qwen_drop4 | UltraBERT minus 4 sliding layers (20L) | qwen pool | 0.82 / 0.881 |
| soup of 3 (qwen, qwen_s1, comb) | | | 0.855 / 0.919 |
| **soup of all 6** (frozen) | | | 0.845 / 0.925 torch; **0.85 int8** |

A prediction-average ensemble of 4 models scored 0.86 / 0.932, so ensembling gains only about one point.

**Latency and quantisation** (gte numbers are from untrained probes; all are ORT on 4 threads):
- gte-modernbert 22L, stock HF export: int8 p95 175 ms on dev. At 384 tokens, 31% of time went to `Where` and 21% to the matmuls.
- With the lean export, the 12-layer probe at 256 tokens went from 72 to 40 ms p50, and the 22-layer probe reached p95 ≈ 81–83 ms on
  dev (machine loaded).
- UltraBERT lean int8 at max_len 384: p95 84 ms (loaded). At max_len 224 with left truncation: 65–75 ms; the final soup measured
  67–69 ms.
- Plain int8 on UltraBERT collapses dev accuracy to 0.515 (v1 saw the same). The cause is the MLP `down_proj` activation outliers:
  keeping only `down_proj` in fp32 gives 0.85 (97% agreement with fp32). SmoothQuant α=0.8 on `down_proj` gives 0.84–0.86 with all
  weights int8. On gte, int8 also lost 4 pts (0.74 to 0.70) for the same reason (MLP `Wo`); SmoothQuant α=0.8 fixed it there too.
- RoPE implemented as slice/concat vs as a matmul with a fixed 128×128 rotation: no clear difference.

## What didn't work / caveats
- gte-modernbert is clearly worse than UltraBERT for BoolQ reading comprehension at this data size.
- More LLM-generated questions (a 3.6x data multiplier) did not help UltraBERT (within noise) and hurt gte. The generated questions are
  passage-first and too easy compared with real BoolQ queries, so I stopped at 3,000 passages and did not spend the rest of the budget
  on them.
- A contrast/paraphrase augmentation pilot over the real questions (100 passages, $0.004) produced visibly noisy "flipped" answers, so I
  did not scale it.
- Seed variance (0.82–0.85 for the same recipe) is larger than most of the data effects, so the soup was the most reliable gain.
- Dropping 4 layers to buy latency cost about 2–3 pts AUROC.

## Library features to merge
- `experiments/boolq/fast_export.py`: lean ONNX export for ModernBERT and UltraBERT, plus `smooth_mlp` SmoothQuant.
- `experiments/boolq/quant.py`: int8 on weight MatMuls only.
- Together these make UltraBERT int8 usable and give about 1.8x CPU speed over the stock export. Should replace `export_onnx` and
  `quantize_int8` for these backbones.
- `experiments/boolq/trainer.py`: layer truncation/dropping and left truncation, via a wrapper around `load_backbone`.
- `experiments/boolq/soup.py`: weight soup.

## Next
- Cross-fitted ensemble distillation: teachers trained on fold A label fold B, plus the generated questions, then one student.
- A larger soup (more seeds).
- Contrast sets verified by a second labeler.
- Static int8 (QDQ) to cut the remaining latency.
- Realistically a 150M encoder trained on ~9k LLM-labeled examples tops out around 0.85 here. Jev's 0.913 looks out of reach for this model
  size with these levers (in v1, gold-only UltraBERT reached 0.821).


## Round 2 (data-side; same backbone and CPU gate)

### Baseline noise
I re-ran the round-1 single-model recipe (UltraBERT, hard qwen labels) with 5 seeds. Dev accuracy was 0.845 / 0.82 / 0.83 / 0.795 /
0.895: mean **0.837**, sd 0.037; mean AUROC 0.922. A single run tells you almost nothing, so every round-2 idea was compared on seed
means.

### Ideas tested (dev, n=200)

| idea | dev result | outcome |
|---|---|---|
| (a) **Soft vote labels**: P(yes) = mean of qwen + (v1 deepseek-seeded, or deepseek-flash plain on the 4.35k texts without v1 labels) + gemini tie-break on 714. Soft BCE. The deepseek-flash labels cost $0.22. | 5 seeds: 0.87 / 0.875 / 0.86 / 0.885 / 0.90; mean **0.878** (sd 0.015), AUROC 0.940 | **success**: +4 pts and less seed variance |
| (b) **Real-style queries**: query written from a 12-word topic hint only, in the style of the 20 seed questions, then answered from the passage by qwen and deepseek-flash (yes/no/unknown); kept only when both agree. 2,500 passages: 19.8k queries, 5.97k kept, 68% "unknown". Cost $0.36 including the pilot. | Added to the hard qwen pool: 3 seeds 0.825 / 0.81 / 0.85, mean 0.828 | **failure**: the queries look like BoolQ, but did not help |
| (c) More LLM-written passages | not run: the budget went to (a), (b) and (d) | n/a |
| Weight soup of soft-label seeds | soup of 3, uncalibrated: 0.78; Platt on silver: 0.85; soup of 5: 0.86 | **failure**: soft-label seeds land in different basins |
| Prediction ensemble of the 5 soft seeds | 0.90 / AUROC 0.959 | best on dev, but 5x latency, so not deployable |
| (d) **Distillation**: the 5-seed soft ensemble labels 29.7k synthetic questions it never trained on (5.97k real-style + 23.7k round-1 generated), target = 0.5·teacher + 0.5·LLM. Student trained on pool_soft + these. | 3 seeds: 0.895 / 0.875 / 0.875, mean **0.882**, AUROC 0.947 | neutral vs (a) on accuracy, +0.7 AUROC |
| Soup of the 3 distilled students + Platt on silver val | 0.88 torch; **0.885 int8**, AUROC 0.947 | success: unlike the raw soft seeds, students that share a teacher soup well |
| int8 variants on a soft model (α 0.6 / 0.8 / 0.9; fp32 down_proj) | 0.85–0.855 vs fp32 0.865 | neutral; kept α=0.8 |

Cross-fitted distillation on the pool texts was not run: the teacher seeds had trained on all of the pool. Only synthetic questions
the teachers never saw got teacher labels.

### Frozen round-2 recipe
All 9.2k pool texts with soft vote labels, plus 29.7k synthetic questions with teacher+LLM soft labels. UltraBERT, lr 1e-5, max_len
224 left-truncated, 2 epochs, 3 seeds. Weight soup of the 3 seeds, Platt-recalibrated on the silver early-stopping split (no gold).
Lean ONNX + SmoothQuant α=0.8 + int8 weight MatMuls. Threshold 0.5. Model: `data/v2/boolq/models/soup_dist3c`.

### Dev comparison that justified the re-score
- Recipe level: seed means went from 0.837 (5 seeds) to 0.878 (soft, 5 seeds) and 0.882 (distilled, 3 seeds).
- Frozen models: int8 dev **0.885 vs 0.85**.
  - Paired bootstrap on accuracy: +3.5 pts, 95% CI [−0.5, +7.5], P(≤0) = 0.05.
  - AUROC: 0.947 vs 0.924, P(≤0) = 0.015.
  - McNemar: 12 vs 5 discordant examples, p = 0.14.
- On the frozen-model accuracy alone the evidence is borderline. The seed-mean gain (about 2 baseline sd) and the AUROC gain are what
  carried it.

### Test (re-scored once, `--rerun-reason` logged)
**0.855** (CI 0.843–0.867), up from 0.844 in round 1; Jev 0.913. The dev gain (+3.5) shrank to +1.1 on test, so dev overstated the
improvement. That is consistent with the dev noise (±2.5 pts) and with dev having been looked at many times.

**Latency:** p50 41.2 ms, p95 64.3 ms (repeats 60.7–64.3; load average ~5).
**Spend:** round 2 $0.58 (deepseek-flash labels $0.22, real-style queries $0.36 including the pilot). Total $1.12 of $1.34.

### Takeaways and next steps
- Soft multi-labeler votes are the one clear round-2 win: they raise the mean and cut seed variance.
- Synthetic questions, even BoolQ-style ones, still do not help when trained on directly with hard labels. They may help a little as
  a distillation carrier.
- The remaining gap to Jev (about 6 pts on test) is large. The 5-model prediction ensemble reached 0.90 on dev, so better distillation
  of that ensemble into one model looks like the most promising next step:
  - cross-fitted teachers on the pool;
  - more unlabeled passages, for which (c) LLM-written passages are the cheap source.


## Round 3 (distilling the 0.90 ensemble into one model; GPU work)

Target: the prediction average of the 5 round-2 soft-label models reached 0.90 on dev (AUROC 0.959). Could one deployable model get
there?

| experiment | dev (n=200) | outcome |
|---|---|---|
| 5-fold cross-fit soft-label teachers on the pool (each on 4/5 of pool_soft). Every pool text gets an out-of-fold (OOF) probability. | Folds 0.83 / 0.875 / 0.84 / 0.865 / 0.875, mean 0.857. OOF probability agrees with the 3-labeler vote on 82% of texts. | input to the students |
| Real-style queries on 1,000 more pool passages, answered by qwen only | 7,952 queries, 3,175 kept, $0.05 | input |
| 10-teacher ensemble (5 full-pool + 5 fold teachers) labels 32.9k synthetic questions | teacher–LLM agreement 0.82 | input |
| **OOF distillation**: pool target = 0.5·OOF + 0.5·vote; synthetic target = 0.5·teacher + 0.5·LLM; 41.7k examples, 3 seeds from pretrained | 0.87 / 0.87 / 0.895, mean 0.878, AUROC 0.945. Soup of 3 (Platt): 0.87. | failure: not above round 2 (mean 0.882) |
| **Shared-init students**: 3 seeds continued from r11_oof_s0 (head included), lr 5e-6, 1 epoch, then weight soup | 0.875 / 0.885 / 0.88, mean 0.880. Soup: 0.875, AUROC 0.950. | neutral |

The shared init did soup cleanly: the Platt slope needed after averaging was 1.5, against 5.3 for separately initialised runs (the soup
barely compresses the logits). It still brought no accuracy gain.

**Dev comparison vs the round-2 frozen model** (soup_dist3c, torch 0.88 / AUROC 0.952):
- Best round-3 candidate (soup_si3c): −0.5 pt accuracy, paired bootstrap 95% CI [−3.5, +2.5].
- Seed means: round 2 distilled 0.882; round 3 OOF 0.878; round 3 shared-init 0.880.
- No round-3 student closed the gap to the 0.90 ensemble.

**Decision: test NOT re-scored.** The CI includes 0, so the round-2 frozen recipe stands: test 0.855, CPU p95 64.3 ms.

**Spend:** round 3 $0.05 (qwen-only real-style queries). Total $1.17 of $1.42. About 4 GPU-hours.

**Takeaways:**
- Every distillation variant converges to about 0.88 on dev:
  - soft vote labels alone;
  - in-sample teachers on synthetic questions;
  - OOF teachers on the pool;
  - shared-init soups.
- Across these students, the target scheme matters less than the student's capacity (UltraBERT-150M at 224 tokens).
- The ensemble's extra ~2 pts are not transferable through the soft targets we can produce. More synthetic data stopped helping after
  round 2.
- Closing a ~6 pt test gap to Jev would need something beyond these data levers, within the fixed backbone size and CPU gate.


## Relaxed setting (steps 2 and 3: 8 CPU threads, backbones up to ~400M), round 4

ModernBERT-large was not tried: it needs ~100 ms at 128 tokens on 8 threads, and BoolQ inputs need ~224. I used the extra threads to
try a prediction ensemble of my 150M UltraBERT models instead.

### Latency (8 threads, `benchmarks.v2.latency --threads 8`, loaded machine)

| what | p95 (load avg) |
|---|---|
| Frozen single model (soup_dist3c) | 48.6 ms (13); 53.5 ms (1). The same run on 4 threads: 63.9 ms. |
| 2 members as separate sessions (the script loops over model dirs) | 253 ms; 3 members: 334 ms |
| 2 members exported as ONE ONNX graph (`experiments/boolq/ens_export.py`: mean of member probabilities; each member gets the lean export + SmoothQuant) | 94 / 92 ms at max_len 224; 74.8 ms at max_len 160 (load ~10) |

- **Separate sessions:** the sessions' spinning intra-op threads fight over the cores. That is 4.7x slower than one model.
- **One graph:** removes the contention, but on one model 8 threads give only ~1.3x over 4, so 2 members are ~2x the compute. That
  misses the 70 ms gate even at 160 tokens.

### Dev accuracy of ensembles (int8 members, 11 candidates)
- **Averages:** all pairs 0.886; all triples 0.888; single models 0.875. Frozen round-2 model: 0.885.
- **Selected on the full dev set:** the best pair reaches 0.92. That is selection bias: the pair was picked by its score on the same
  200 examples it is judged on.
- **Honest split-half selection** (pick the combination on one half, score it on the other, vs the frozen model on that half):

  | | pick on A, score on B | pick on B, score on A |
  |---|---|---|
  | pair | 0.88 vs 0.87 | 0.90 vs 0.90 |
  | triple | 0.92 vs 0.87 (CI [+1, +10]) | 0.88 vs 0.90 |

  Mixed results, within noise.
- **One-graph pair (dev, single-request inference):** 0.895–0.90 vs 0.89–0.895 for the frozen model.

### Longer max_len
The frozen model with left truncation scored 0.89 / 0.895 / 0.89 / 0.89 at max_len 192 / 224 / 256 / 384. Truncation costs nothing,
so longer inputs gain nothing.

### Decision
**No relaxed re-score.**
- Ensembles do not pass the gate on 8 threads.
- Their honestly selected dev gain is within noise.
- Longer inputs do not help.

The relaxed-setting result for BoolQ is therefore the same as the strict round-2 result: **0.855 test, p95 64.3 ms (4 threads) / ~49–54
ms (8 threads)**. No LLM spend in this step; the total is still $1.17.

A shared-trunk multi-head ensemble (one encoder, several heads) would fit the gate. But the members' diversity comes from full-backbone
seeds, so it probably would not keep the ensemble gain. That is the one idea left on this axis.
