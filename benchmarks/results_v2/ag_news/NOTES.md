# AG News (v2 re-certification)

**Result (test, scored once):** accuracy **0.893** (95% CI 0.886-0.900) vs Jev **0.885** on the same 7,600 examples. Beats Jev.
**CPU latency (ONNX int8, 4 threads, batch 1, shared busy machine):** p50 29 ms, **p95 51 ms** (< 70 ms gate).
**Label spend:** $0.027 (dev-only labeler comparisons; training labels reused from v1 at no cost).

## Final recipe (FROZEN.json)
- Labels: reused v1 deepseek-v4.1-flash labels from the **plain** prompt (`data/silver/ag_news/silver.jsonl`), `gold` field ignored,
  v2 seed/dev uids dropped -> 4,970 texts (90/10 split: 4,473 train, 497 held-out *silver* for early stopping / temperature).
- Student: gte-modernbert, full fine-tune, lr 3e-5, bs 32, max 3 epochs (best epoch 2), max_len 128 (covers the 99th pct of tokens), seed 0.
- Export: ONNX + dynamic int8 (per-channel, MatMul/Gemm). Model: `data/v2/ag_news/models/v1plain_5k/`.

## Dev evidence (200 gold dev; A/B = even/odd halves)
| idea | dev acc |
|---|---|
| labeler deepseek-v4.1-flash, plain prompt (v1 prompt) | 0.905 (A .900 / B .910) |
| labeler + 20 v2 seeds | 0.905 (.880 / .930) |
| labeler + v2 seeds, soft distribution | 0.910 |
| student on v1 plain labels, 5k | **0.920** (torch fp32) |
| student on v1 plain labels, 2.5k | 0.925 |
| student on v1 *seeded* labels, 5k (v1 used different seeds drawn from the pool) | 0.900 |
| final model after int8 quantisation | 0.915 (argmax agrees with fp32 on 97.5% of dev) |

- Learning curve 2.5k -> 5k is flat on dev, so no new pool labels were bought (the 120k pool was not needed).
- Prompt variants do not change labeler agreement beyond noise (SE on 200 dev ~ 2 pts); plain labels were kept.

## Notes / next
- Margin over Jev is small (+0.8 pt, CI lower bound just above Jev). Things that might add a point: ensembling 2 seeds
  (latency budget allows ~2 int8 models at ~25 ms p50 each, but p95 would approach the gate), more data with a stronger labeler
  on the student/labeler disagreements, or int8-aware choice of seed. Not tried, to stay within the budget and one-shot protocol.
- Shared helper code for all three wins_choice tasks lives in `experiments/ag_news/` (oyj_common.py, train_student.py,
  export_check.py, finalize_probs.py, quant_variants.py, label_dev_base.py). No library changes were needed.
