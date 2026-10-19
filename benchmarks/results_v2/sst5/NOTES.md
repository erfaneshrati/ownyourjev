# SST-5 (v2): notes

## Result (test, scored once with final_eval)
- **Ours: argmax accuracy 0.583 (95% CI 0.562 to 0.603); Jev on the same 2,210 examples: 0.579.** That beats Jev by 0.004, a gap well inside the CI, so read it as parity.
- Spearman: ours 0.852, Jev 0.851.
- CPU latency (ONNX Runtime, 4 threads, batch 1, partial int8): p50 21.4 ms, **p95 29.4 ms**. It passes the 70 ms gate. It was measured while the machine was busy (load average around 30), so the real numbers are lower.
- Label spend: **$0.94** (scope sst5). Test was scored once and never re-run.

## Final recipe (frozen on dev)
1. **Labeler.** deepseek/deepseek-v4.1-flash, temperature 0, `distribution=True` (a probability for every level). The prompt has the task spec, SST-convention guidance (`experiments/sst5/guidance_v1.txt`) and the 20 gold seeds as few-shot examples. All 8,324 pool texts were labeled (minus seeds and dev).
2. **Student.** UltraBERT-150M, full fine-tune, trained on the soft targets (`soft_labels=True`). Settings: lr 2e-5, 3 epochs, batch size 32, max_len 128, seed 1. Early stopping and temperature were fitted on 400 held-out silver examples, so no gold was used in training. The decision is a plain argmax with no bias calibration.
3. **Deployment.** ONNX with partial dynamic int8. Every MatMul is quantised except the MLP `down_proj`, the attention-score MatMuls (QK^T and AV) and the head; those stay fp32. Code: `experiments/sst5/partial_q.py`, variant `nodown`. The model is in `data/v2/sst5/models/final/model.int8.onnx`.

## What we tried on dev (200 gold; SE about 0.035, so differences under about 0.03 are noise)
### Labelers (20 seeds, soft labels)
| labeler | no guidance | guidance v1 |
|---|---|---|
| deepseek-v4.1-flash | 0.530 | **0.620** |
| qwen3.7-flash | 0.550 | 0.595 |
| gemini-2.5-flash-lite | 0.540 | 0.580 |
| claude-haiku-5.5 | 0.560 | 0.555 |
| deepseek-v4-pro | n/a | 0.615 (cost $0.15 for 200 texts) |
- v1's deepseek-flash labels scored 0.547 (plain) and 0.573 (seeded) on the 117 dev items they cover.
- **The biggest single lever was the guidance.** Without it, the labelers almost never used "very positive": 32 of 34 gold 4s came out as 3. The guidance says that clear, enthusiastic praise is a 4 even when the snippet is short. It also says that ordinary criticism is a 1 rather than a 0, gives rules for what counts as neutral or mixed, and asks for probability to be spread over adjacent levels. I wrote it from the errors on dev half A only. Half B, which I did not look at, still improved (0.53 to 0.57).
- Averaging the deepseek and qwen labels did not help (0.595). Blending v1 seeded labels at weight 0.2 gave +0.017 on 117 items (noise), so I left it out.
- Per-class log-bias calibration on the labeler, estimated with 5-fold CV on dev, helped haiku (0.56 to 0.60) and deepseek+guidance (0.62 to 0.64). It did not help the students (below).

### Students (all trained on the deepseek-flash guidance-v1 soft labels unless noted)
- gte-modernbert, soft labels, 9 runs over seeds, lr in {2e-5, 3e-5, 5e-5} and 3 to 5 epochs: 0.540 to 0.585, mean about 0.56. Run-to-run noise is about ±0.02, even for the same seed (GPU nondeterminism).
- gte-modernbert with hard labels: 0.550. Hard v1 seeded silver: 0.535. Mean pooling: 0.570.
- **UltraBERT, soft labels: 0.600, 0.615, 0.610 (lr 2e-5, seeds 0, 1, 2) and 0.610 (lr 1e-5). It beat gte-modernbert in every run (+0.04 to +0.05 on average) and had higher Spearman (0.86 to 0.87).**
- Ensembles of two UltraBERTs averaged 0.614 against 0.609 for single models, which is noise, so I deployed a single model.
- Bias calibration on the student (CV on dev, or matching the teacher's argmax on silver val): no consistent gain (between -0.03 and +0.03).
- Adding the 220 gold examples (×3) to training, cross-fitted over 2 dev folds with gte: 0.570. That is no clear gain over the 0.56 mean.
- 3,500 extra SST-style sub-phrases split from pool sentences and labeled by the teacher ($0.17): gte with 3 seeds scored 0.555, 0.550 and 0.545. No gain; Spearman went up slightly (0.855 to 0.859).

### Quantisation and latency (dev accuracy after ONNX export)
- gte-modernbert: fp32 ONNX 0.545 at p95 50 ms. Full int8 dropped it to 0.495 (−0.05) at p95 26 ms, so on this task int8 is not free for gte either.
- UltraBERT: full int8 collapses it (0.600 to 0.310). fp32 ONNX had p95 89 ms (under load) and torch had p95 439 ms, both failing the gate.
  - **Partial int8 that keeps `down_proj` and the attention-score MatMuls in fp32** gave 0.590 (ub_s0) and 0.600 (ub_s1), with p95 27 to 30 ms. The fp32 model scored 0.615, so the loss is about 0.01 (2 examples).
  - Quantising only the attention projections gave 0.610 but p95 71 ms, which fails the gate.
- The deployed model's dev result is acc 0.600 and Spearman 0.857. I chose seed 1 because it had the best single-model dev score (0.615 fp32), which is a mild selection effect.

## Label spend ($0.94 total)
- Dev labeler comparisons: about $0.36, of which deepseek-v4-pro was $0.15.
- Full-pool deepseek-flash labels: $0.43.
- Phrases: $0.17.
- I reused the v1 silver labels only for dev comparisons and one discarded student. Those v1 seeded labels used different seed examples drawn from the training pool.

## Library notes (for merging)
- **Partial dynamic quantisation** (`nodes_to_exclude` = every `mlp/down_proj` MatMul, `self_attn/MatMul*` and the head) makes int8 work for UltraBERT: −0.01 instead of −0.29 dev accuracy, at about 3× the speed of fp32. This could become an option in `export.quantize_int8`. It may also fix the BoolQ int8 problem.
- The student training wrapper (`experiments/sst5/train_student.py`) averages soft labels from several sources and can add extra transfer texts and cross-fitted gold folds.

## What I would try next
- Cross-fitted gold added to UltraBERT training, and a seed or checkpoint average folded into one model.
- Self-consistency samples to sharpen the teacher distributions. The teacher scores 0.62 on dev and the student about 0.61, so the teacher is now the ceiling.
- Labeler bias calibration applied to the soft labels before distillation (it gave +0.02 CV on dev for the labeler).

## Round 2 (cap raised to $1.54; total spend now $1.22, of which round 2 was $0.29)
Every experiment is logged in `experiments.jsonl` (20 round-1 lines backfilled, 11 round-2 lines). Dev set: 200 examples. Half A is the half I used to write the guidance; half B is held out.

| idea | dev result | baseline | outcome |
|---|---|---|---|
| Guidance v2, written from the remaining labeler errors on half A | 0.615 (A 0.71, B 0.52) | v1: 0.620 (A 0.67, B 0.57) | failure: overfits A, −0.05 on held-out B |
| Self-consistency: 4 samples at T=0.7, averaged | samples 0.56 to 0.585; average 0.595 | greedy T=0: 0.620 | failure |
| Batch labeling, 20 texts per call | flash 0.560, v4-pro 0.575, haiku 0.530 / 0.565 | per-item: 0.620 / 0.615 / 0.560 | about 8× cheaper, but −0.03 to −0.06 accuracy per labeler |
| Two labelers averaged (per-item flash + one other) | flash+pro 0.640, flash+haiku 0.645, flash+batched haiku 0.635 | 0.620 | noise-level (+0.02) |
| Full pool relabeled by batched haiku ($0.17); UltraBERT trained on flash+haiku mean (3 seeds) | 0.625 / 0.555 / 0.575, mean 0.585 | round-1 recipe, 5 seeds: 0.600 / 0.615 / 0.610 / 0.610 / 0.580, mean 0.603 | failure |
| Weight soup of 3 UltraBERT seeds, head averaged | 0.525 (Spearman 0.863) | 0.603 | failure: the averaged head breaks class 3 |
| Soup of 3 seeds, head refit on silver soft labels (LR, no gold) | 0.625 fp32 | 0.603 | success, but noise-level |
| Soup of 5 seeds + refit head, partial int8 | 0.620 fp32, **0.605 int8** | round-1 frozen model: 0.600 int8 | neutral |

- Boundary-class targeting: the guidance v2 rules for mixed versus negative and for mockery versus criticism are the boundary-class attempt. They did not generalise to half B.
- Pool coverage was already complete in round 1 (all 8,324 texts were labeled).

**Decision: no test re-score.** The best round-2 model (5-seed soup, partial int8) beats the round-1 frozen model by +0.005 on dev. A paired bootstrap on dev gives a 95% CI of [−0.040, +0.050], with P(diff ≤ 0) = 0.46. The fp32 soup beats the fp32 mean over seeds by +0.017, which is also inside the noise. By the policy, the round-1 test result stands: 0.583 vs Jev 0.579, parity.

**Why SST-5 is stuck near 0.60.** The teacher's agreement with gold is the ceiling: deepseek-flash with guidance is about 0.62, and students reach 0.60 to 0.62. Every labeler I tried, alone or averaged, lands between 0.60 and 0.645 on dev, and a 200-example dev set cannot tell those apart. SST-5 gold is itself noisy: one seed rates "a wacky and inspired little film" as neutral. Getting clearly past Jev (0.579) would most likely need a teacher well above 0.65. None of the affordable labelers gets there.

**Library candidates.** `experiments/sst5/soup_full.py` (seed soup + head refit on silver) removes best-of-seed selection at no inference cost. `experiments/sst5/batchlab.py` (batched soft labeling) cuts cost about 8×, but loses accuracy on this fine-grained task.

## Round 3: stronger-labeler probe (cap $3.02; total spend $2.49, of which round 3 was $1.27)
Setup: every candidate was run per item, with guidance v1, the 20 seeds and soft labels. I judged them on dev half B (100 examples, held out when the guidance was written) against the current labeler, deepseek-v4.1-flash with guidance v1, which scores 0.570 on B. The paired bootstrap uses 5,000 resamples. The current models (Gemini 3.x, Claude Sonnet 5.5) do not allow reasoning to be turned off, so they ran at effort "low". gpt-5.6-terra needed concurrency 3 to avoid rate limits.

| labeler | reasoning | B accuracy | diff vs flash (95% CI) | Spearman | cost (100 items) |
|---|---|---|---|---|---|
| deepseek-v4.1-flash (current) | off | 0.570 | n/a | n/a | n/a |
| gpt-5-mini | low | **0.600** | +0.03 [−0.05, +0.11] | 0.824 | $0.049 |
| deepseek-v4.1-flash | medium | 0.590 | +0.02 [−0.05, +0.09] | 0.817 | $0.038 |
| claude-sonnet-5.5 | low | 0.580 | +0.01 [−0.07, +0.09] | 0.841 | $0.456 |
| gemini-3.1-pro-preview | low | 0.580 | +0.01 [−0.08, +0.10] | 0.829 | $0.429 |
| qwen3.7-plus | off | 0.570 | +0.00 [−0.09, +0.09] | 0.819 | $0.019 |
| gpt-5.6-luna | off | 0.560 | −0.01 [−0.09, +0.07] | 0.827 | $0.009 |
| gemini-3.5-flash-lite | low | 0.560 | −0.01 [−0.09, +0.07] | 0.815 | $0.051 |
| gpt-5.6-terra | low | 0.540 | −0.03 [−0.12, +0.05] | 0.828 | $0.086 |
| deepseek-v4-pro | medium | 0.520 | −0.05 [−0.14, +0.03] | 0.814 | $0.127 |

Averaging labelers on B: flash + gemini-pro 0.610, flash + gpt-5-mini 0.610, all five 0.620 (+0.05, CI [−0.02, +0.12]).

**Decision: stop. No pool relabel, no retrain, no test re-score.** No labeler reached the bar (≥0.66, or a paired CI excluding 0). Frontier models (Sonnet 5.5, Gemini 3.1 Pro, GPT-5.6 Terra) do no better than the cheap labeler on SST-5 gold, and reasoning does not help (deepseek-v4-pro with medium reasoning was the worst). Frontier labeling also costs about $0.004 per item, so relabeling the uncertain part of the pool would cover only a few hundred texts within the cap.

For the paper: on SST-5, LLM agreement with gold plateaus at about 0.56–0.60 per held-out half for every model, from $0.03/M to frontier. The ceiling is the dataset's annotation noise and conventions, not labeler strength. The round-1 test result (0.583 vs Jev 0.579, parity) stands.

## Round 4: relaxed setting (8 threads, backbones up to ~400M; no new LLM spend)
- Dev token lengths (ModernBERT tokenizer): p95 is 50 tokens and the maximum is 64, so I used max_len 80.

| idea | dev accuracy | compared with | outcome |
|---|---|---|---|
| ModernBERT-large (395M) on soft deepseek-flash labels, 4 runs (lr 2e-5 seeds 0, 1, 2; lr 1e-5) | 0.580 / 0.615 / 0.575 / 0.580, mean **0.588** | UltraBERT-150M mean over 5 seeds: 0.603 | failure: the larger student does not exceed its labels |
| ModernBERT-large on flash + batched-haiku label mean | 0.580 | same seed on flash only: 0.580 | failure |
| ModernBERT-large seed 1 with the library's int8 | 0.605 (fp32 0.615) | n/a | p95 145 ms at 8 threads under load (load average ~25): fails the gate |
| Two-model UltraBERT partial-int8 ensembles (all 10 pairs of 5 seeds) | mean 0.594 | single models: 0.596 | failure; p95 56 ms at 8 threads |
| Shared-init soup: UltraBERT seed 1 continued for 1 epoch with 4 data orders, then weight-averaged | fp32 **0.625**, partial int8 0.600 | round-1 frozen: 0.600 (int8) | fp32 diff +0.025, CI [−0.01, +0.06]; int8 diff 0.000 |
| fp32 ONNX UltraBERT at 8 threads | n/a | n/a | p95 90 ms (load average 13): fails the gate |

- Round-1 frozen model at 8 threads: p95 28.3 ms.
- The frontier probe labels exist only for dev, so they cannot be used as distillation targets. Only the pool-wide labels were available: deepseek-flash, batched haiku and v1 silver.

**Decision: no re-score.** No deployable model beats the round-1 frozen model on dev beyond noise. The best int8 candidate ties it at 0.600. The only noticeable gain, the fp32 shared-init soup (+0.025, CI includes 0), comes from fp32 weights that miss the latency gate.

Bigger students, ensembles and soups all land at 0.59–0.62. That matches the teacher's dev agreement (0.62): on SST-5 the student is capped by its labels, not by model capacity. The strict round-1 result (test 0.583 vs Jev 0.579) remains the result in both settings.
