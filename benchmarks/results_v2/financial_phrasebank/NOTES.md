# Financial PhraseBank (v2 re-certification)

**Result (test, scored once):** accuracy **0.794** (95% CI 0.770-0.819) vs Jev **0.731** on the same 970 examples. Beats Jev.
(v1 reported 0.840 on test with a recipe picked after seeing test scores; the clean v2 number is lower.)
**CPU latency (ONNX int8, 4 threads):** p50 19 ms, **p95 32 ms** (< 70 ms gate).
**Label spend:** $0.023 (dev-only labeler/prompt comparisons; training labels reused from v1).

## Final recipe (FROZEN.json)
- Labels: reused v1 deepseek-v4.1-flash labels from the v1 **seeded** prompt (`silver_seeded.jsonl`; v1's 20 seed examples were
  drawn from the training pool, different from the v2 seeds), gold ignored, v2 seed/dev uids dropped -> 3,637 texts.
- Student: gte-modernbert, full fine-tune, lr 3e-5, bs 32, max 3 epochs, max_len 96 (covers all texts), seed 1,
  early stopping on 10% held-out silver.
- Served as ONNX dynamic int8 (per-channel). Model: `data/v2/financial_phrasebank/models/v1seeded_s1/`.

## Dev evidence (200 gold dev)
Labeler agreement (deepseek-v4.1-flash): plain prompt 0.850 (v1 labels, cached), v2 seeds 0.845, v2 seeds soft 0.820,
guidance G1 (positive-leaning rules written from dev-half-A errors: most errors were gold positive -> labeled neutral) 0.845,
G1 + seeds 0.850 (B half 0.840 vs 0.890 for plain). Guidance did not help -> no new labels bought.
v1 seeded labels on dev uids agreed 0.799 with gold.

Students (fp32 torch):
| labels | seed 0 | seed 1 | mean |
|---|---|---|---|
| v1 plain | 0.840 | 0.850 | 0.845 |
| v1 seeded | 0.850 | 0.860 | 0.855 |
| v1 plain∩seeded agreement filter (3,221) | 0.865 | 0.845 | 0.855 |
All within noise; seeded labels picked (best mean, no filtering), seed-1 model taken (mild optimistic dev bias; mean is the honest estimate).

Quantisation (this model, dev): fp32 ONNX 0.860; int8 per-channel s8 0.850; u8 0.820; per-tensor 0.805; reduce_range 0.830.
int8 flips ~15% of argmax decisions on dev (low-margin neutral/positive borderline cases) but accuracy changes little.
fp32 ONNX p95 was 57 ms then 69 ms on the busy machine, too close to the gate, so int8 was frozen (dev 0.850).

## Next
- int8 instability: quantisation-aware fine-tuning or picking the seed whose int8 agrees best with fp32 on dev.
- Dev -> test drop (0.85 -> 0.79) suggests the test split's label distribution/agreement level differs (the dataset's
  test split mixes annotator-agreement levels); soft/agreement-weighted training could help.
