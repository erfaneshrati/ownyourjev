# HelpSteer2 (v2): notes

**Result (round 1, scored once):** mean Spearman over the five attributes **0.314 vs Jev 0.412** on the same 1,038
test examples. We do **not** beat Jev. CPU latency p95 **59.2 ms** (p50 50.4) on 4 threads, ONNX int8, all five heads in
one encoder pass, so it is under the 70 ms gate. The machine was shared, and two of four runs were hit by contention bursts
(p95 140 ms). See Latency.

| attribute | ours (test) | 95% CI | Jev (same examples) | dev estimate (5-fold CV) |
|---|---|---|---|---|
| helpfulness | 0.142 | 0.079–0.202 | 0.400 | 0.291 |
| correctness | 0.146 | 0.083–0.205 | 0.369 | 0.305 |
| coherence | 0.204 | 0.144–0.262 | 0.274 | 0.210 |
| complexity | **0.524** | 0.480–0.571 | 0.460 | 0.570 |
| verbosity | 0.554 | 0.507–0.598 | 0.558 | 0.564 |
| **mean** | **0.314** | | **0.412** | 0.388 |

Complexity beats Jev, and verbosity ties it. Coherence is behind, and helpfulness and correctness are far behind. The
student cannot judge whether a response is factually right or follows the prompt's instructions. This is where the gap
is. On helpfulness and correctness the dev estimate (about 0.29–0.30) did not carry over to test (0.14): see "Dev→test gap".

Label spend: **$1.444** of the $1.50 scope. The experiment ledger (22 entries) is in `experiments.jsonl`.

## Final recipe (frozen in FROZEN.json)
- **Model:** one `gte-modernbert-base` encoder (all 22 layers), CLS pooling, and a linear layer with 5×5 outputs
  (5 score heads). A second set of 5 auxiliary heads exists in the checkpoint but is not deployed. It was trained only on
  gold rows here and is sliced away. Code: `experiments/helpsteer2/mh.py` and `train_dual.py`.
- **Input (latency-driven):** `[CLS] "prompt <#tok> response <#tok> lines <#lines>" [SEP] prompt [SEP] response [SEP]`,
  128 tokens in total. The prompt gets up to 30% of the room (or more if the response is short), taken as 1/3 head + 2/3
  tail. The response gets the rest (2/3 head + 1/3 tail). The length header lets verbosity and complexity see the full
  length even though the text is truncated.
- **Data:** 7,338 deepseek-v4.1-flash hard labels, from the v1 prompt with no seeds: 4,938 reused v1 silver (the v1
  plain-prompt labels, after dropping the 42 uids in seeds/dev) plus 2,400 new pool labels. We also trained on the 220
  gold examples (200 dev + 20 seeds) with weight 5. Targets are discretised Gaussians (sigma 0.6) around each level.
- **Training:** 3 epochs, AdamW, lr 3e-5 (head 5x), batch 32, linear warmup/decay, bf16, seed 0. The model is the one
  after epoch 3, with no selection on dev.
- **Post-processing:** the helpfulness and correctness heads both output the mean of their two expected scores, as a
  2-point distribution. The two gold labels correlate at r = 0.95, so averaging two noisy estimates of nearly the same
  thing helps. This gave +0.01 on every model and on both dev halves.
- **Quantisation:** ONNX dynamic int8, per-channel. `mlp.Wo` of layers 4, 9, 11 and 15, and the head, stay fp32.
- **Artifacts:** `data/v2/helpsteer2/models/FINAL_r1/` holds `model.int8.onnx`, `post.json`, `eval_<attr>.npy` and
  `dev_probs_final.npy`.

## What we tried (dev = 200 gold examples; mean Spearman of the 5 attributes unless noted)
Dev noise is large. Spearman on 200 examples has an SE of about 0.07 per attribute and about 0.03 on the mean. The two
dev halves differed by up to 0.3 on helpfulness for the same labeler.

**Labelers (agreement of the labeler itself with gold dev):**
- deepseek-v4.1-flash hard 0.398, soft 0.419, soft + 20 seeds 0.430, soft + seeds + HelpSteer2 rubric guidance 0.408.
  The rubric did not help.
- qwen3.7-flash soft + seeds + guide 0.364; gemini-2.5-flash-lite 0.204 (poor).
- Compact "decimal rating" prompt (one fractional 0–4 score per attribute; `label_dec.py`): flash 0.403, qwen 0.379,
  qwen + seeds 0.425. The qwen decimal prompt costs 4.5e-5 per call, a third of flash hard.
- Labeler ensembles: flash_hard + qwen_dec 0.435; adding qwen_dec_seeds gives 0.463. The ensemble gain is real at the
  labeler level, but it did **not** carry over to the student (next section).
- Cost lesson: the client's default `provider.sort=throughput` route charged about 1.4e-4 per flash call.
  `provider.sort=price` charged 5.3e-5 for the same model and prompt.

**Student (L22, 128 tokens unless noted):**

| idea | dev | vs | outcome |
|---|---|---|---|
| v1 silver 4.9k flash, 5 heads, length header (3 seeds) | 0.352 / 0.356 / 0.346 | v1 test 0.334 | baseline |
| one-hot targets instead of sigma 0.6 | 0.342 | 0.352 | neutral |
| depth vs tokens: L11@192 / L8@256 / L6@320 | 0.330 / 0.346 / 0.336 | 0.352 | depth matters more than tokens |
| rule-based degraded pairs (mismatched response → h/c 0; truncated response → lower), +1.5k | 0.317 | 0.352 | **failure** |
| qwen decimal labels instead of flash (same 4.9k) | 0.290 / 0.278 | 0.356 / 0.351 | **failure** |
| flash+qwen merged targets (4.9k) | 0.343 / 0.327 | 0.356 / 0.351 | failure |
| merged labels on 15k (mostly qwen-only) | 0.335 / 0.341 | 0.356 | failure |
| labeler-specific heads (flash heads + qwen auxiliary heads, 7.3k flash + 20k qwen) | 0.366 / 0.347 | 0.364 / 0.363 | neutral |
| +2.4k flash labels (7.3k flash) | 0.364 / 0.363 | 0.356 / 0.351 | success (+0.01) |
| 5 epochs (best epoch 2) | 0.352 / 0.356 | 0.364 | 3 epochs best |
| L18 (latency margin) | 0.319 / 0.330 | 0.364 / 0.363 | −0.04, rejected |
| ridge stacking of the 5 heads (CV on dev) | 0.335 | 0.355 | failure (too few dev examples) |
| h/c mean blend (no fitted parameters) | 0.373 / 0.376 | 0.364 / 0.363 | success |
| + gold dev/seeds in training, weight 5 (5-fold CV) | **0.388** | 0.373 / 0.376 | +0.012–0.015 (paired bootstrap CI −0.02…+0.06), gains in h/c |
| gold weight 20 (5-fold CV) | 0.382 | 0.388 | neutral |

The main surprise was that **the labeler's agreement with gold does not predict how good the student gets.** qwen's
dev agreement was close to flash's (0.38 vs 0.40), and the flash+qwen ensemble was clearly better (0.435). Yet students
trained on qwen labels scored 0.28–0.29, against 0.35 for students trained on flash labels. Merging the two labelers
also hurt. One possibility is that flash's judgements follow features a small encoder can read from 128 tokens, while
qwen's do not. We have not tested this.

**Quantisation:**
- Quantising every weight MatMul to int8 cost 0.04 dev (0.355 → 0.315; agreement with fp32 0.84–0.94 per attribute).
- Keeping `mlp.Wo` in fp32 restored fp32 accuracy (0.360, agreement 0.997).
- A per-layer sweep found the activation outliers concentrated in `mlp.Wo` of layers 4, 9, 11 and 15. Single-layer
  agreement there was 0.92–0.985, against 0.993 everywhere else. Keeping only those four fp32 gave agreement 0.988–0.993
  and dev 0.358 vs 0.360, and it is about 4 ms faster. That is the deployed variant.

**Latency (batch 1, 4 threads, int8, shared machine):**
- Full inputs are far over the gate. The median input is 430 tokens and p95 is about 1,000. At 22 layers, p95 is about
  140 ms at 256 tokens and about 550 ms at 512.
- Below about 128 tokens, latency is dominated by per-layer overhead. At 22 layers, 96/112/128 tokens gave p95
  61/66/68 ms. So only removing layers buys latency, and removing layers costs accuracy (L18: −0.04).
- Things that did not help: the ORT transformer optimizer (it halves the node count but was not faster) and session
  options (dynamic_block_base, denormals-as-zero).
- Final pipeline (`experiments/helpsteer2/latency_mh.py`): tokenise, truncate, forward and softmax for all 5 heads,
  plus the h/c blend, on dev states, n=100 after 3 warm-up requests.
  - p50/p95 by run, in ms: 50.4/59.2 at load 5.4 (reported); 50.7/61.1 at load 13.7; 62.0/139.3 (n=300) and
    117.1/140.2 during contention bursts.
  - Cross-check with the official `benchmarks.v2.latency --onnx` on a view of the same graph
    (`FINAL_r1_official_view`: right-truncated 128 tokens, no header): 52.8/62.2 ms.
  - The official script cannot run our custom truncation and header, so we mirror its protocol in `latency_mh.py`.
  - Please re-measure on a quiet machine. The margin under 70 ms is about 10 ms.

## Dev→test gap (helpfulness/correctness)
The dev CV estimate was 0.29/0.31 on helpfulness and correctness, but test came out at 0.14/0.15. Complexity,
verbosity and coherence carried over (0.57→0.52, 0.56→0.55, 0.21→0.20).

We checked one possible leak after the test run. 196 of the 200 dev prompts have their sibling response in the training
pool, so the student might recognise the prompt rather than judge the response. We split dev into examples whose
sibling was flash-labelled in training (n=105) and examples without one (n=95). There was **no** effect: 0.378 vs 0.372,
0.365 vs 0.388, 0.378 vs 0.383. So the gap is most likely dev noise (SE about 0.07 per attribute at n=200, plus
optimistic selection over many variants) and/or a shift between the train and validation splits. It is not leakage.

The honest reading is that a 150M encoder seeing 128 tokens does not learn the "is it correct, does it follow the
instructions" notion from about 7k LLM labels. The LLM labelers themselves reach about 0.35–0.40 on dev for these
attributes.

## Deviations, implementation notes (mergeable)
- `experiments/helpsteer2/fastbudget.py` (`FastBudget`, `FastCache`) are drop-in replacements for the ledger and cache
  objects. Large labeling jobs crawled at about 1 call/s for two reasons:
  - Every call commits to the shared SQLite files from inside the asyncio loop while many agents write to them.
  - Every call runs a full-table `SUM` scan of a ledger with 190k rows.
  The wrappers keep the same caps and write the same rows. They batch the writes in a background thread every 2 s and
  reuse the ledger totals for 5 s plus any spend recorded locally since then. Suggest merging into `ownyourjev.llm`.
- `mh.py`: a shared-encoder multi-head model with response-preserving truncation and a length header, ONNX export with
  selective int8, and an end-to-end ONNX predictor. A multi-question model is what any multi-head task needs.
- `train_dual.py`: labeler-specific heads with masked losses, gold rows with weights, CV folds over dev, and fixed-epoch
  selection.
- Seeds were used only to measure labelers and as training rows in the final model. The v1 silver labels are the plain
  v1 prompt; we did not use the seeded v1 labels.

## What I would try next (needs budget)
- **Flash labels on the rest of the pool:** 12.8k more labels via the cheap route would cost about **$0.68**. Going from
  4.9k to 7.3k flash labels gave +0.01 on dev, so I expect another +0.01–0.02. That is below the dev noise, so it is not
  enough on its own to justify a round 2.
- **A stronger labeler for helpfulness/correctness only** (e.g. deepseek-v4-pro with reasoning, or self-consistency
  over flash samples) on about 5k examples: roughly **$1.5–2.5**. The bottleneck is that the student cannot detect
  factual and instruction errors. Better labels on those two attributes is the one lever that might move them. The
  labeler-vs-student mismatch we saw with qwen suggests checking student transfer on 1k examples before scaling.
- **Pairwise training on the 9.9k same-prompt pairs:** it costs no LLM spend, since the pairs only need the existing
  labels. A ranking loss on within-prompt differences is implemented (`train_mh.py --rank_w`) but was not evaluated
  because GPU time ran out.
- **Latency headroom** for a deeper or longer input would need a smaller per-layer overhead, for example a static-shape
  graph or fused attention for ModernBERT in ORT. The token budget is not the binding factor below 128 tokens.

## Round 2 (more LLM budget; scope cap $3.44): no recipe beat round 1 on dev, so test was **not** re-scored
Round-2 spend: **$0.856** (total for the task: $2.300). Ledger entries are tagged round=2.

**1. Stronger labeler for helpfulness/correctness only** (2-question prompt, no seeds). Each probe labelled the 200 dev
examples. Spearman with gold (help / corr) and cost per call:

| labeler | help | corr | $/call |
|---|---|---|---|
| deepseek-v4.1-flash, no reasoning | 0.388 | 0.410 | 2.7e-5 |
| deepseek-v4.1-flash, reasoning (low effort) | **0.458** | **0.439** | 1.2e-4 |
| deepseek-v4.1-flash, reasoning (medium effort) | 0.484 | 0.426 | 1.1e-4 |
| deepseek-v4-pro, no reasoning | 0.364 | 0.355 | 1.1e-4 |
| claude-haiku-5.5 | 0.423 | 0.354 | 1.2e-4 |
| claude-haiku-5.5, reasoning | 0.389 | 0.379 | 2.6e-4 |
| gemini-3.1-flash-lite, reasoning | 0.459 | 0.402 | 3.7e-4 |

Jev on test scores 0.400 / 0.369. Frontier models (sonnet-5.5 at $2/$10 per M tokens) would cost about $2.4 per 1k
examples, which was out of budget.

- **Student probe** (as the coordinator asked, before scaling): flash-with-reasoning labelled 1.5k pool examples
  ($0.167). We trained students only on those rows, once with the reasoning h/c labels and once with the original flash
  h/c labels for the same rows, 3 seeds each. Mean5 per seed: 0.308 (reasoning) vs 0.310 (original). Paired bootstrap
  on mean5: −0.002 [−0.036, +0.030]. On h/c alone: −0.017 [−0.086, +0.054]. **No student gain**, so we did not label more.
- **512-token diagnostic** (ignoring latency), same probe, 2 seeds: 0.315 vs 0.312, paired +0.004 [−0.032, +0.043].
  More context does not make the better labels learnable at this scale either.
- Same pattern as qwen in round 1. Labels that agree better with gold do not give a better student on these attributes:
  the encoder does not learn factual and instruction checking from a few thousand LLM labels.

**2. Pairwise ranking loss** on same-prompt pairs (1,376 flash-labelled pairs; `train_dual.py --rank_w`), 2 seeds,
fixed 3 epochs. With rank_w 1.0, mean5 was 0.336/0.364; with 0.3, 0.332/0.343. The baseline was 0.364/0.363.
Failure.

**3. Flash labels on the rest of the pool** (12,766 more at 3.6e-5/call, $0.464): 20.1k flash labels in total. Students
without gold, 2 seeds: 0.360 vs 0.374 for the 7.3k set. Paired: −0.016 [−0.058, +0.024]; on h/c: −0.049
[−0.135, +0.038]. Complexity and verbosity went slightly up, h/c down. Failure. We stopped the gold-CV version of this
recipe because the paired comparison already ruled out a dev win.

**Round-2 conclusion:** the round-1 frozen recipe stands: test 0.314 vs Jev 0.412, latency p95 59 ms. Under the
128-token / 4-thread gate, LLM-label quality is not the lever for helpfulness/correctness. What remains is model
capacity and context: bigger backbones or 8 threads (the coordinator's next steps), or a different kind of supervision
that targets errors.

## Round 3, relaxed setting (8 threads): no gain, so test was **not** re-scored
- **Latency at 8 threads** (round-1 model, int8, batch 1, i9-10900X with 10 physical cores, shared). 8 threads is only
  10–20% faster than 4 at batch 1. Raw-ONNX p95 swung a lot with machine load:
  - 128 tokens: 43–67 ms.
  - 144 tokens: 75 ms in one loaded run.
  - 160 tokens: 54–84 ms.
  - 192 tokens: 63–103 ms.
  - 256 tokens: 97–151 ms.
  So 192 tokens fits the gate only when the machine is quiet. The deployed round-1 pipeline end-to-end at 8 threads
  measured p50 41.3 / p95 44.7 ms (`latency_relaxed_8threads.json`).
- **Cheapest relaxed variant** (the round-1 recipe at 192 tokens, 3 seeds, no gold) vs 128 tokens (3 seeds): mean5 0.343
  vs 0.367. Paired bootstrap: −0.024 [−0.051, +0.003]; on h/c: −0.054 [−0.106, −0.004]. **Worse.** This is consistent
  with the round-2 512-token probe. No relaxed re-scoring; the round-1 result (0.314 vs Jev 0.412) stands, and no
  relaxed-setting row beats Jev.
- **Why ModernBERT-large was not tried:** the coordinator's measurement puts it at about 100 ms p95 at 128 tokens on 8
  threads, which is over the gate for these inputs.
