# Emotion (v2 re-certification)

**Result (test, scored once):** accuracy **0.812** (95% CI 0.795-0.829) vs Jev **0.585** on the same 2,000 examples. Beats Jev by 22.6 pts
(v1: 0.638 with test-selected variants).
**CPU latency (fp32 ONNX, 4 threads, busy shared machine):** p50 31 ms, **p95 44 ms** (< 70 ms gate).
**Label spend:** $0.379 (dev prompt/labeler comparisons ~$0.10 incl. one $0.05 Haiku probe; 10,000 pool labels ~$0.28).

## Key finding: labeling guidance
The dair-ai/emotion gold label is the emotion category of the word the author attaches to "feel/feeling" (hashtag/word-derived),
not the overall mood: "i am not feeling shitty anymore" -> sadness, "i love running because i feel strong" -> joy,
"he makes me feel so greedy" -> anger. The plain labeler reasons about overall mood, which caps it at ~0.5.
Guidance G1 (`experiments/emotion/guidance.py`) states the rule (find the feel-word, ignore negation/context and other emotion words,
use the key adjective inside "feel like ...") plus a word->category list. It was written from the 20 seeds and the labeler's
errors on dev half A only; half B is the held-out check.

## Dev evidence (200 gold dev; A = half used to write guidance, B = held-out half)
| labeler variant (deepseek-v4.1-flash unless noted) | dev | A | B |
|---|---|---|---|
| plain prompt (v1) | 0.490 | 0.450 | 0.530 |
| + 20 v2 seeds | 0.540 | 0.480 | 0.600 |
| + 20 v2 seeds, soft distribution | 0.510 | 0.520 | 0.500 |
| G1 guidance, no seeds | 0.765 | 0.830 | 0.700 |
| **G1 + seeds** (chosen) | **0.785** | 0.840 | **0.730** |
| G1 + seeds, qwen3.7-flash | 0.745 | 0.790 | 0.700 |
| G1 + seeds, claude-haiku-5.5 (7.5x the cost) | 0.805 | 0.850 | 0.760 |
v1 labels on the 71 dev uids v1 had labeled: plain 0.39, seeded 0.48.
Haiku was +2-3 pts but would have cost ~$2.50 for 10k labels; not affordable in the shared $0.60 cap.

| student (gte-modernbert, full FT, max_len 96) | dev acc (torch fp32) |
|---|---|
| v1 plain labels 5k | 0.535 |
| v1 seeded labels 5k | 0.600 |
| G1+seeds labels 2.5k | 0.735 |
| G1+seeds labels 5k, seed 0 / seed 1 / 5 epochs | 0.795 / 0.780 / 0.795 (early stopping picks epoch 2 either way) |
| **G1+seeds labels 10k**, seed 0 / seed 1 | **0.800** / 0.790 |
The steep 2.5k -> 5k curve justified buying 5k more labels; 5k -> 10k gave only +0.5 pt (the student is at the labeler's ceiling).

Quantisation: int8 (dynamic, per-channel) lowers dev 0.80 -> 0.78 (argmax agreement 93.5%) on both the 5k and 10k models,
so the fp32 ONNX export was frozen (p95 44-45 ms, int8 would be 22 ms).

## Final recipe (FROZEN.json)
G1 guidance + 20 v2 seeds, deepseek-v4.1-flash, 10,000 pool texts (`data/v2/emotion/labels/pool_g1_seeds_10000.jsonl`; the first 4,903
are the uids v1 had labeled, rest random) -> gte-modernbert full FT (lr 3e-5, bs 32, 3 epochs max, max_len 96, seed 0, early stopping on
10% held-out silver) -> fp32 ONNX. Model: `data/v2/emotion/models/g1_10k/` (served from `g1_10k_fp32/`, symlinks without the int8 file,
because OnnxPredictor prefers model.int8.onnx when present).

## Not tried / next
- Soft labels with G1 (soft seeded labels lowered agreement 0.54 -> 0.51 and double output cost).
- Haiku for disagreement relabeling only (student vs labeler disagreements, ~10-15% of the pool) would cost ~$0.3.
- Guidance iteration 2: remaining A errors are mostly genuine label noise (e.g. "passionate" labeled joy and love), so little headroom.
