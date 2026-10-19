# toxigen v2: notes

**Result (test, scored once):** `toxic` accuracy **0.854** (95% CI 0.830–0.878) vs Jev **0.878** on the same 940 examples.
**Does not beat Jev.** `toxicity` Spearman **0.790** (CI 0.761–0.814) vs Jev **0.840**. Test AUROC for toxic was 0.924 (Jev 0.949).
**CPU latency:** p95 **36.9 ms**, p50 30.7 ms (one shared encoder answers both heads; fp32 ONNX, 4 threads, busy machine). Passes the 70 ms gate.
**Label spend:** $0.728 in scope `toxigen` ($0.20 on dev experiments, $0.52 on labeling the full pool). The v1 silver labels were reused at no cost.
No test reruns.

## Final recipe (frozen on dev)
- Labels: deepseek-v4.1-flash, no seeds, with labeling guidance v1 (`experiments/toxigen/common.py:GUIDANCE_V1`, which covers implicit
  or coded hate versus merely mentioning a group, and condescension or "positive" stereotypes as mild). Both questions are asked in
  one call with `distribution=True`, which returns P(toxic) and a 5-level distribution. All 8,740 pool texts were labeled (65 failures,
  dropped). Seed and dev uids were excluded.
- Mixed in the v1 silver labels (`silver.jsonl` and `silver_seeded.jsonl`, both deepseek-v4.1-flash, hard labels) where available, at
  weight 0.5 against the new soft labels (the v1 seeded labels used different seed examples drawn from the training pool). The v1 `gold`
  field was ignored, and dev/seed uids were dropped.
- Student: `experiments/toxigen/mt_model.py:MultiHead`. It is gte-modernbert-base with CLS pooling and one `Linear(768, 6)`: column 0 is
  the toxic logit (BCE on soft P), columns 1–5 are toxicity (soft cross-entropy). Full fine-tune for 3 epochs, lr 3e-5 (head ×5),
  bs 32, max_len 64 (max token length in the pool is 45). It is a uniform weight soup of 3 seeds.
- `toxic` probability = E[toxicity]/4, with threshold **0.30** (the maximum of the 7-point smoothed dev accuracy curve).
  `toxicity` = the 5-level softmax.
- Runtime: fp32 ONNX (`data/v2/toxigen/models/soup_silv05_s012/model.onnx`). Its 6-column `probs` output loads through
  `OnnxPredictor` and `benchmarks.v2.latency` unchanged.

## Dev evidence (200 gold; "best" = threshold tuned on the same dev, optimistic; "cv" = 2-fold threshold CV)
Labelers, soft distribution prompt. Columns are acc(E[score] at best threshold) / Spearman / AUROC(E[score]):
| variant | acc_best | Spearman | AUROC | cost/200 |
|---|---|---|---|---|
| flash plain | 0.885 | 0.719 | 0.913 | $0.019 |
| flash + 20 gold seeds (soft seed answers) | 0.910 | 0.766 | 0.951 | $0.012 |
| flash + guidance v1 (**chosen**) | 0.905 | 0.754 | 0.957 | $0.012 |
| flash + seeds + guidance | 0.895 | 0.769 | 0.930 | $0.016 |
| flash, hard labels, seeds + guidance | 0.89 | 0.718 | 0.881 | $0.010 |
| deepseek-v4-pro + seeds + guidance | 0.905 | 0.795 | 0.946 | $0.022 |
| claude-haiku-5.5 + seeds + guidance | 0.92 | 0.764 | 0.955 | $0.062 |
| gemini-2.5-flash-lite + seeds + guidance | 0.885 | 0.708 | 0.931 | $0.043 |
| qwen3.7-flash + seeds + guidance | 0.875 | 0.656 | 0.811 | $0.007 |
- Labeler ensembles helped on dev: flash_g1+haiku reached 0.93 / 0.787 / 0.973, and 4-way ensembles reached Spearman ~0.80.
  Labeling the pool with haiku or pro, though, would have cost $0.9–2.7 against the $1 cap. Cheap second labelers (qwen,
  gemini-lite) added nothing. v1 silver hard labels agree with dev gold 0.85 (114 overlapping dev uids).
- The LLM's own P(toxic) ranks worse than its E[toxicity] (AUROC 0.79–0.95 vs 0.91–0.96). Deriving `toxic` from the toxicity
  distribution fits the gold binarisation (toxicity_ai + toxicity_human > 5.5). On dev that binarisation is exactly
  human mean ≥ 3 of 5.
- Students (gte-modernbert, 3 epochs, epoch-3 dev numbers):
  | run | Spearman | AUROC(es) | acc_es best / cv | acc_p best / cv |
  |---|---|---|---|---|
  | v1 silver seeded only (4.9k, hard) | 0.757 | 0.950 | 0.89 / 0.85 | 0.885 / 0.88 |
  | v1 silver plain+seeded | 0.742 | 0.943 | 0.87 / 0.835 | 0.88 / 0.85 |
  | flash_g1 soft (8.7k) seed 0 | 0.773 | 0.958 | 0.92 / 0.915 | 0.905 / 0.90 |
  | same, seed 1 | 0.762 | 0.954 | 0.90 / 0.86 | 0.90 / 0.86 |
  | + silver 0.5, seed 0 | 0.770 | 0.960 | 0.915 / 0.91 | 0.915 / 0.905 |
  | + silver 0.5, seed 1 / seed 2 | 0.761 / 0.754 | 0.956 / 0.959 | 0.90 / 0.91 | 0.905 / 0.895 |
  | lr 2e-5 | 0.775 | 0.958 | 0.91 / 0.885 | 0.905 / 0.86 |
  | 2 epochs | 0.777 | 0.959 | 0.915 / 0.91 | 0.90 / 0.86 |
  | **soup of the 3 silver-0.5 seeds (final)** | 0.770 | 0.958 | 0.905 (0.89 at the frozen 0.30) | 0.90 |
  Full-pool soft flash_g1 labels beat v1 silver (AUROC +0.01–0.015, Spearman +0.02). Silver mixing, lr and epochs were all within
  seed noise (about ±0.02 accuracy). E[toxicity] beat the toxic head on threshold-CV accuracy (mean 0.896 vs 0.877 over 5 runs).
  The student matches or beats its labeler on dev (Spearman 0.77 vs 0.754).
- Quantisation: int8 dynamic (per-channel) hurt clearly. On seed 0, dev AUROC went 0.960 → 0.940, Spearman 0.769 → 0.729,
  best accuracy 0.915 → 0.89. fp32 ONNX already passes the gate (p95 37–41 ms), so int8 was not used. (int8 p95 was about 24 ms.)

## What didn't transfer
Dev looked like a win: toxic accuracy 0.89–0.905 and AUROC 0.958. On test the model scored 0.854 accuracy and AUROC 0.924.
Dev overstated the model, beyond the threshold optimism. A 200-example dev set (57 positives) has an accuracy CI of about ±0.04,
and the test split seems harder for this labeler and student. The ceiling is the labeler: flash's dev agreement is 0.905 at best,
with Spearman 0.75. Jev's Spearman of 0.84 is above every single labeler I could afford.

## Library notes (for merging)
- `experiments/toxigen/fast_budget.py:FastBudget`. `Budget.check` runs two full-table SUMs (~150 ms on the shared ledger) synchronously
  inside the asyncio loop for every request. One 8.7k gather stalled for minutes before sending anything, and a single process topped
  out at ~50–60 calls/min. FastBudget caches the sums for 10 s and adds local spend. Even so, the synchronous sqlite cache/ledger work
  limited throughput, so the pool was labeled in 6 parallel processes (`label_pool.py SHARD 6`, then `merge`).
  Worth moving those calls to a thread or caching them in the library.
- `mt_model.py:MultiHead` is a shared encoder with K heads in one Linear and an ONNX export whose `probs` holds all heads.
  It is a candidate for a library multi-question model.
- OpenRouter models priced above $0.20/M prompt tokens (gpt-5.6-luna, gemini-3.1-flash-lite) fail because of the client's
  `max_price` provider cap.

## Next
- A stronger or ensembled labeler for the pool (haiku + flash, or pro) needs about $1–3. On dev it is worth +0.02–0.03 accuracy and
  +0.03–0.05 Spearman, which is the most likely way to close the gap.
- Use the 200 dev gold in training through cross-fitting, upweighted, so the student learns the exact binarisation.
- A larger dev set would help, or at least treat dev gains under ±0.04 as noise.

## Round 2 (data side: labeler ensemble). Not re-scored on test.
Spend in round 2: $1.38. Scope total is $2.11 of the $2.33 cap. Round-1 experiments were backfilled into `experiments.jsonl`
(round=1), and every round-2 experiment is logged there (round=2).

**Batched labeling** (`experiments/toxigen/blabel.py`): 20 texts per call, and each reply gives P(toxic) plus a 5-level
distribution per text. Dev numbers (acc at best threshold / Spearman / AUROC of E[toxicity]):
- flash + guidance: 0.92 / 0.765 / 0.961
- flash + guidance + soft gold seeds: 0.91 / 0.788 / 0.950
- haiku + guidance: 0.895 / 0.797 / 0.948
- haiku + guidance + seeds: 0.895 / 0.753 / 0.937 (seeds don't help haiku)
- gemini-flash-lite: 140 of 200 replies invalid. deepseek-v4-pro and gpt-5-mini: OpenRouter 404.

Pitfall: the first attempt set max_tokens to 40 per text, so replies were truncated. The truncated replies were *cached*,
because the cache key ignores max_tokens, so I had to change the prompt text before re-running.

**Chosen ensemble** (mean of the 4 soft label sets: round-1 flash_g1 + batched flash, flash+seeds and haiku). On dev it scores
0.92 / **0.818** / 0.972, against flash_g1 alone at 0.905 / 0.754 / 0.957. Labeling the 8,740-text pool with the three batched
labelers cost $1.26: flash $0.49 and $0.50, haiku $0.27. Flash was pricier than the dev estimate because of provider routing.

**Student on the ensemble targets.** Same architecture and recipe as round 1, without silver, 3 seeds, then a soup.
| | round 1 (frozen) | round 2 | paired bootstrap of the difference (2000 resamples) |
|---|---|---|---|
| AUROC(E[tox]) soup | 0.9583 | 0.9625 | +0.004, 95% CI [−0.001, +0.011] |
| Spearman soup | 0.7699 | 0.7755 | +0.006, 95% CI [−0.009, +0.019] |
| toxic acc, threshold by 2-fold CV (50 random splits) | 0.878 | 0.892 | +0.014; new better in 86% of splits |
| seed means (Spearman / AUROC) | 0.762 / 0.958 | 0.769 / 0.962 | |

The label-level gain (+0.064 Spearman) mostly does not reach the student. Both bootstrap CIs include 0, and +0.014 CV accuracy
is well inside the ±0.04 dev noise. **Per the re-scoring policy, test was not re-scored.** The round-1 result stands:
test accuracy 0.854 vs Jev 0.878.

Other round-2 ideas (seed 0, on the ensemble labels):
- An auxiliary MSE loss on expected toxicity gave Spearman 0.769 / AUROC 0.961. Neutral.
- 5 epochs gave 0.772 / 0.963. Neutral.
- **Transductive** training: the scored texts, with ensemble soft labels (no gold), were added to training. On dev as a stand-in
  for eval_inputs, ×1 gave 0.791 / 0.968 and ×3 gave **0.796 / 0.972**. This is the only clear dev gain: the student moves toward
  the ensemble labeler on exactly the texts it is scored on. I did **not** adopt it. It needs the LLM ensemble to have seen the
  test texts, so it says nothing about quality on unseen traffic, and it would weaken a "replace Jev" claim. The protocol
  technically permits it, since eval_inputs are unlabeled. Applying it would cost about $0.15 to label the 940 eval_inputs with
  the 3 batched labelers, and it is the coordinator's or user's call.

Next options, all needing more budget:
- A stronger single labeler (pro or gpt-5-mini through a working route).
- Synthetic implicit-hate texts labeled by the ensemble, to give more in-distribution training data.
- Cross-fitted training on the 200 dev gold examples, upweighted.

## Round 3, relaxed setting (8 threads, backbones up to ~400M). Not re-scored on test.
Same setup as round 2 except the backbone. I trained the shared-encoder, two-head model on **modernbert-large** (395M; lr 2e-5,
3 epochs, max_len 64) with the round-2 targets (mean of the 4-labeler ensemble's soft labels), then made a soup of 3 seeds.
`toxic` is still derived from E[toxicity], with the threshold chosen by CV on dev. No LLM spend.

| model (dev, n=200) | AUROC(E[tox]) | Spearman | CV accuracy (50 splits) |
|---|---|---|---|
| round-1 frozen (gte-modernbert-base soup) | 0.9583 | 0.7699 | 0.878 |
| round-2 base on ensemble labels | 0.9625 | 0.7755 | 0.892 |
| **large soup**, fp32 | 0.9604 | 0.7760 | 0.877 |
| large soup, int8 | 0.9596 | 0.7774 | n/a |
| base (r2) + large (int8), averaged | 0.9645 | 0.7808 | 0.881 |

Large soup vs round-1, paired bootstrap of the difference:
- AUROC: +0.002, 95% CI [−0.013, +0.018]
- Spearman: +0.006, 95% CI [−0.029, +0.042]
- CV accuracy: −0.001

Individual large seeds scored 0.770–0.777 Spearman and 0.957–0.961 AUROC, the same as the 150M seeds. **Conclusion: ToxiGen is
capped by labels and data, not by student capacity.** A student 2.6× larger gains nothing. The base + large ensemble is also
within noise, so nothing was frozen and test was **not** re-scored.

- int8 on modernbert-large is lossless here, unlike gte-base in round 1.
- Latency at 8 threads, on a heavily loaded machine (load average ~18–23 on 20 cores):
  - large int8: p50 48 ms, p95 84.5 ms. That fails the gate under this load, and would need a re-check on a quiet machine.
  - round-1 base fp32: p50 29 ms, p95 35.6 ms.
