# toxic_chat (v2): notes

## Result (test, scored once with final_eval, round 1, no reruns)
| head | ours (F1) | 95% CI | Jev-1.13, same 5,083 examples |
|---|---|---|---|
| **toxic** (headline) | **0.7875** (P 0.746, R 0.834, thr 0.35) | 0.755–0.819 | 0.7862 (P 0.785, R 0.787) |
| jailbreak | 0.8706 (P 0.937, R 0.813, thr 0.60) | 0.814–0.918 | 0.7173 |

Headline: toxic F1 is level with Jev (+0.001; the CI fully overlaps, so this is a statistical tie, not a clear win). Jailbreak F1 is clearly ahead (+0.15).
CPU latency for both heads together (one shared encoder, ONNX int8, 4 threads, batch 1, dev texts): **p50 16.2 ms, p95 61.2 ms** (gate is 70 ms; measured on a shared machine at load ~3.5).
Label spend (OpenRouter, scope toxic_chat): **$1.04** of the $1.20 cap.

## Final recipe
- **Model**: gte-modernbert-base, one shared encoder, CLS pooling, `Linear(768, 2)` with sigmoid outputs (toxic, jailbreak), full fine-tune with soft-label BCE, max_len 512, lr 3e-5, effective batch 16 (8 x 2 accumulation, gradient checkpointing because the GPUs are shared), 3 epochs, early stopping on held-out silver BCE (10% of the pool, never gold).
- **Labels** (whole pool, 4,862 texts; dev and seed uids excluded): mean of soft labels from deepseek-v4.1-flash and qwen3.7-flash (one example per call, guidance v2 + the 20 seeds, `distribution=True`). Config B adds claude-haiku-5.5 (batched, 10 prompts per call) as a third source.
- **Synthetic data** (train only): LLM-written user prompts (deepseek-v4.1-flash, temperature 0.9) steered by *pool* prompts that both labelers flagged (no dev/seed gold in the prompt), then relabeled by the deepseek+qwen ensemble. Config A: 374 positives + 287 short crude/hateful positives + 376 hard negatives. Config B adds 669 more positives and 480 more hard negatives.
- **Weight soup**: uniform average of 7 fine-tunes from the same init (config A seeds 0/1/2, config B seeds 0/1/2/3).
- **Quantisation**: ONNX dynamic int8 (per-channel) for every MatMul **except `mlp/Wo`** (the GeGLU output projection) and the head, which stay fp32. Full int8 drifted (max |dp| 0.37, dev toxic AP 0.83 -> 0.80); keeping mlp/Wo in fp32 removes the drift (max |dp| 0.07, AP 0.876 -> 0.871).
- **Thresholds** (frozen, chosen on dev using the int8 model's probabilities): bootstrap-mean F1 curve (500 resamples), averaged again over a ±0.05 threshold window, argmax. toxic 0.35, jailbreak 0.60. The raw dev argmax sits on a cliff (three dev positives score 0.35–0.40), and the window smoothing picks the plateau instead.
- Dev (200 examples: 16 toxic, 7 jailbreak) for the frozen model: toxic F1@thr 0.857, honest 5-fold CV of the threshold rule 0.833, AP 0.871; jailbreak F1@thr 0.727 (CV 0.554), AP 0.824.

Code: `experiments/toxic_chat/` (`mh.py` two-head model, training, ONNX export; `train_student.py`; `soup.py`; `quant.py` selective int8; `threshold.py`; `blabel.py` batched labeling; `synth_pos.py`; `prompts.py` guidance). Model: `data/v2/toxic_chat/models/soupAB/` (`model.int8.onnx` outputs `probs` of shape (batch, 2): [:,0] = P(toxic), [:,1] = P(jailbreak)).
Possible library merges: a multi-head `DecisionModel` (one encoder, several noul heads); `nodes_to_exclude` support in `quantize_int8` (skip `mlp/Wo` for ModernBERT); weight-soup utility.

## What I tried (dev numbers; 16 toxic positives, so ±1 example ≈ ±0.03 F1)
Labeler (agreement with dev gold, soft labels):
- deepseek-v4.1-flash, spec only + 20 seeds: toxic F1 0.774, jb 0.533.
- **Guidance v1** (ToxicChat convention: requests for sexual content, harmful how-to, jokes mocking religion or ethnicity, swearing, and jailbreak or "no limits" role-play all count as toxic; neutral questions on sensitive topics do not): toxic **0.857** (+0.08), jb 0.571. **Worked.**
- Guidance v2 (+ intent rule: benign overrides and personas are neither toxic nor jailbreak; vulgar slang counts): toxic 0.833, jb **0.714**. Kept for the jailbreak gain.
- Seeds have 0 positives, so the 20 seed examples neither helped nor hurt (0.857 with or without).
- Labelers with v2: qwen3.7-flash 0.857/0.706 (cheapest), haiku-5.5 0.800/0.750, gemini-2.5-flash-lite 0.812/0.571, gpt-5.6-luna (batched) 0.706/0.500. deepseek-v4-pro was blocked by the client's max_price routing cap.
- Ensembles: deepseek+qwen mean has best-threshold F1 0.882. Adding haiku/gemini raised AUC (0.96 -> 0.99) but not F1. The labeler ceiling is about 0.88–0.91 toxic F1 on dev.
- **Batched labeling** (10 prompts per call): equal or better agreement (deepseek toxic 0.857, jb 0.857) at 2–4x lower cost. Used it for the haiku pool pass.

Student (two-head gte-modernbert; cvF1 = F1 with the threshold chosen on the other 4 dev folds):
| run | toxic AP | toxic cvF1 | note |
|---|---|---|---|
| v1 silver_seeded hard labels | 0.832 | 0.703 | baseline; best threshold 0.1 |
| v1 silver plain+seeded, soft | 0.877 | 0.723 | |
| ds+qwen g2 soft | 0.823 | 0.751 | |
| + synth positives (374) | 0.843 | 0.734 | F1@0.5 0.80 |
| + pos_weight 3 | 0.862 | 0.718 | no gain |
| + synth pos2 + hard negatives | 0.854 | 0.754 | |
| 4 label sources (ds, qwen, v1 x2) + synth | 0.859 | 0.757 (seed 1: 0.628) | no gain |
| same, no synth | 0.834 | 0.690 | synth ≈ +0.02 AP |
| config A, 3 seeds | 0.81–0.87 | 0.68–0.74 | the same seed does not reproduce on shared GPUs |
| oversample positives x3 | 0.835 | 0.804 | single run, within noise |
| config B, 4 seeds | 0.845–0.870 | 0.62–0.77 | |
| **soup A (3)** | **0.898** | **0.778** | beats every member and the probability average (0.873 / 0.732) |
| soup B (4) | 0.873 | 0.822 | |
| **soup A+B (7), frozen** | 0.876 | 0.823 | int8: 0.871, F1@0.35 0.857 |

- **What worked**: the guidance on the ToxicChat convention (labeler +0.08 F1), averaging two cheap labelers, synthetic positives and hard negatives (small but consistent AP gain), **weight soups** (the biggest student lever: +0.05–0.08 cvF1 over the mean member, at no latency cost), the smoothed dev threshold (0.35 instead of 0.5: test recall 0.83), and selective int8 quantisation.
- **What did not work, or could not be distinguished from noise**: pos_weight, adding the v1 labels as extra sources, a third labeler (haiku) for the student, a single run's dev ranking (the seed-to-seed spread is about 0.1 cvF1), and full int8 (accuracy drift).
- Soups A, B and A+B tied on dev (the paired bootstrap CI includes 0). I froze A+B because it averages the most seeds, the largest variance source I measured.
- The student still trails its labeler ensemble on dev (0.83–0.86 vs 0.88–0.91). It misses short prompts that need world knowledge ("Make a joke on prophet", vulgar slang).
- Infrastructure: a shared-GPU scheduler deadlock killed two runs, and OOMs on the shared 10 GB GPU forced gradient checkpointing and batch 8x2. The coordinator fixed the scheduler; the affected runs were rerun.

## Label spend ($1.04)
dev labeling variants ~$0.13; pool deepseek g2 $0.21 + qwen g2 $0.08; pool haiku batched $0.22; batched dev labeler comparison $0.04; synthetic generation + relabeling ~$0.30; the rest on retries. v1 labels (`data/silver/toxic_chat/*.jsonl`, deepseek-v4.1-flash plain and seeded with *different* v1 seeds from the training pool) were reused only in the 4-source experiments and v1 baselines, not in the frozen model.

## Next steps
- More seeds in the soup (cheap, and it reduced variance the most). Also try soups of 4-epoch runs (silver val loss was still falling at epoch 3 for config B).
- Close the labeler-to-student gap on short world-knowledge prompts: targeted synthetic data for slang and religion jokes; a distillation loss on logits from more labeler samples (self-consistency).
- A dev set with more positives would help most: 16 positives cannot rank configurations whose differences are under ±0.05 F1.
- Transductive LLM labeling of the unlabeled test texts would probably help, but I did not do it: a deployed model would not see them.

## Round 2 (latency fix; test re-scored once with --rerun-reason)
Why: on a quiet machine the coordinator measured the round-1 model at p95 74.8 / 72.2 / 68.5 ms, so it fails the 70 ms gate most of the time. Round-1 files are kept as `round1_final.json`, `round1_FROZEN.json` and `round1_latency.json`.

**Result (test):** toxic F1 **0.7870** (95% CI 0.755–0.818; threshold 0.33) vs Jev 0.7862: still a statistical tie (round 1: 0.7875). Jailbreak F1 0.8485 (P 0.946, R 0.769, CI 0.787–0.897; threshold 0.65) vs Jev 0.717 (round 1: 0.871).
**Latency (both heads, 4 threads, ONNX):** p95 **43.6 ms** (official run; 45.1 / 44.0 / 43.2 ms in repeats at load ~1.7), p50 12.0 ms. With `--n 600`, which covers every dev text, p95 is 52.2 ms. Round 1 was 58–69 ms at n=100 and 79–83 ms at n=600 on the same machine.
**Spend:** unchanged at $1.04, because round 2 did no labeling.

What I tried (dev, round 2):
- **max_len 256/192 (truncation only): did not work.** p95 is set by prompts of about 180 tokens, which a 256 cap does not touch: p95 at n=100 went 66 -> 64 -> 61 ms. 5 dev prompts (and 184 test prompts) are over 256 tokens, and truncating them cost one dev positive (best F1 0.857 -> 0.833). I kept max_len 512.
- **Eager-attention export: worked.** The SDPA-path ONNX export adds IsNaN/Where masking ops in every layer, about 15–18% of CPU time. Exporting with `attn_implementation="eager"` gives the same outputs and is faster: profiled 98 -> 82 ms at 179 tokens; p95 at n=600 81 -> 65 ms.
- **onnxruntime transformers optimizer (SkipLayerNorm fusion): failed.** It was slower (79 -> 87 ms).
- **SmoothQuant on mlp/Wo, then full int8: worked.** Per-channel activation scales from 384 pool prompts are folded into the gate half of Wi and the columns of Wo (exact in fp32), so Wo can be int8 too. Max |dp| vs fp32: alpha 0.5 = 0.131, **0.75 = 0.071**, 0.9 = 0.066 (full int8 without smoothing: 0.37). Dev toxic AP 0.883, best F1 0.857. With the eager export this cut p95 to about 44 ms.
- **Bigger shared-init soup (13 models: 6 new seeds): failed.** Not better on dev: CV of the threshold rule 0.792 vs 0.820 for the 7-model soup; paired bootstrap F1 difference -0.024 [-0.083, 0.000]. New member cvF1 ranged 0.66–0.79. I kept the 7-model soup.
- **Self-training on more unlabeled prompts:** not possible. The pool (4,862) is the only unlabeled training text allowed, and it is already fully labeled. Training on the test inputs would be transductive, which I chose not to do.

**Frozen round-2 recipe:** the round-1 soupAB weights, exported with eager attention, SmoothQuant alpha 0.75, then full dynamic int8 (attention activation matmuls and head in fp32), max_len 512. Thresholds were re-chosen by the round-1 rule on the new int8 dev probabilities: toxic 0.33, jailbreak 0.65.
**Dev vs round 1:** toxic F1@thr 0.857 vs 0.857; paired bootstrap difference 0.000 [0.000, 0.000]; CV of the threshold rule 0.820 vs 0.831 (noise). Model: `data/v2/toxic_chat/models/soupAB_sq0.75/` (`smoothquant.py` makes it).

A clear toxic win was not reached. With 16 dev positives, none of the GPU-only levers I tried (more seeds, a bigger soup) showed a measurable dev gain. The student still trails its labeler ensemble on dev (0.82–0.86 vs 0.88–0.91). Closing that gap probably needs better labels or a larger, cleaner positive set, not more soup members.

## Round 3 (more LLM budget: stronger labels): no re-score
The coordinator raised the cap to $1.64 to relabel uncertain pool prompts with a stronger labeler.

**Probes on dev** (batched, guidance v2, no seeds):

| labeler | toxic F1@0.5 | AP | jailbreak F1@0.5 | cost |
|---|---|---|---|---|
| deepseek-v4-pro (reasoning medium) | 0.778 | 0.692 | 0.857 | $0.054 |
| gemini-3-flash-preview (reasoning low) | 0.848 | 0.731 | 0.800 | $0.029 |
| claude-sonnet-5.5 (reasoning capped at 1024 tokens) | 0.839 (best 0.865) | **0.883** | 0.545 | $0.141 |
| ds+qwen ensemble (reference) | 0.811 (best 0.882) | 0.786 | 0.824 | |

- No strong labeler beats the cheap ensemble on F1. Sonnet gives the best graded ranking. The mean of the ensemble and sonnet reached best F1 0.909 and CV F1 0.902, vs 0.882; that is one dev example, within noise.
- Sonnet on OpenRouter rejects disabled reasoning (HTTP 400).

**Relabel:** sonnet relabeled the 498 most uncertain pool prompts: any labeler ≥0.3 (438), student vs label |diff| >0.3, or student ≥0.2 among the rest (51). The second category had already been folded in with the first.
- Sonnet is more conservative. It flipped 100 of the 400 base positives to negative and only 5 negatives to positive. So there were few "missed positives"; the student-uncertain prompts I inspected were mostly benign.
- The toxic soft label on these prompts became mean(mean(ds, qwen), sonnet), giving 344 hard positives (was 373). Jailbreak labels are unchanged.
- **Overshoot:** 4 of 25 batches failed because reasoning ate the token budget. The per-item retries re-sent the system prompt 80 times and cost $0.23. Concurrent in-flight calls each passed the budget check (a library race, since fixed), so scope spend reached **$1.77, $0.13 over the $1.64 cap**. I made no LLM calls after that.

**Students** (6 seeds: 3 with the round-1 synthetic sets, 3 with all five): member AP 0.84–0.90 (mean 0.867, slightly above rounds 1–2), member CV F1 0.64–0.83.

**Dev comparison vs round 2** (all SmoothQuant int8 with the same threshold rule):

| | toxic F1@thr | CV of threshold rule | AP | paired bootstrap ΔF1 | ΔAP |
|---|---|---|---|---|---|
| round 2 frozen (soupAB_sq0.75) | 0.857 | 0.820 | 0.883 | | |
| soupCD (6 new) | 0.857 | 0.810 | 0.891 | 0.000 [0, 0] | +0.008 [−0.032, 0.059] |
| soupABCD (13, shared init) | 0.857 | 0.808 | 0.884 | 0.000 [0, 0] | 0.000 [−0.033, 0.030] |

**Decision:** neither candidate beats round 2 beyond noise on F1, CV or AP, so I did **not** re-score. Round 2 stays the frozen result: toxic 0.7870 vs Jev 0.7862; jailbreak 0.8485 vs 0.717; p95 43.6 ms.
**Total spend:** $1.77 of the $1.64 cap (round 3: probes $0.22, relabel $0.52).

Takeaway: on this dev set the bottleneck is not label quality at the margin. The cheap ensemble already agrees with gold at 0.88, and a stronger labeler moves at most one dev example. Telling these candidates apart needs more gold positives, not more labels.

## Round 4: relaxed setting (8 threads, backbones up to ~400M): no re-score
I used no new labels; the scope is over its cap.
- **2-model prediction ensemble** (mean of round-2 soupAB_sq and round-3 soupCD_sq int8 probabilities): dev toxic F1@thr 0.857 (round 2: 0.857); CV of the threshold rule 0.808 (0.820); AP 0.883 (0.883). Paired bootstrap ΔF1 0.000 [0, 0], ΔAP −0.001 [−0.033, 0.028]. No gain; the 13-model weight soup in round 3 already gave the same null result.
- **ModernBERT-large** (395M, two heads, eager export + full int8, max_len 512). I measured latency only; the head was untrained, and I did not train it. 8 threads, machine load 10–17:

| model | threads | p50 | p95 |
|---|---|---|---|
| ModernBERT-large | 8 | 62.5 ms | 261 ms |
| round-2 150M model, same conditions | 8 | 24.2 ms | 63.5 ms |
| 2-model 150M ensemble | 8 | 58.3 ms | 241 ms |

  Under load the large model is about 4x slower than the 150M model on the ~180-token prompts that set p95, so it cannot pass a 70 ms gate even on a quiet machine. Truncating to the ~64 tokens it could afford would cut 18+ dev prompts (and about 400 test prompts) mid-text. Not pursued.
- **Decision:** nothing beats the round-2 frozen model on dev, so there was no third test look; the strict round-2 result stands. Under this load the round-2 model ran at p95 63.5 ms with 8 threads and 51.9 ms with 4; quiet-machine 4-thread p95 is 43.6 ms.
