# Prompt Injections (v2 re-certification)

**Result (test, scored once, round 1):** accuracy **0.957** (95% CI 0.922-0.991) vs Jev **0.741** on the same 116 examples -> beats Jev.
CPU latency (ONNX int8, 4 threads): p50 18 ms, **p95 51.9 ms** at load avg ~12 -> 7 (shared machine; other runs 53 ms at load 33,
77 ms at load 41, 175 ms at load ~40 while one of our own 12-thread jobs ran: tail driven by load, inputs up to 264 tokens).
Label spend: **$0.185** (dev prompt comparisons ~$0.06, pool labels $0.009, synthetic data + relabeling $0.14). Ledger: `experiments.jsonl`.

## Labels (dev 200 gold, halves A/B)
| labeler / prompt | dev acc |
|---|---|
| deepseek-v4.1-flash plain | 0.895 |
| flash + 20 v2 seeds | 0.945 |
| flash + seeds + guidelines (written from the task + the 20 seeds) | **0.965** (A 0.96 / B 0.97) |
| same, soft labels | 0.960 |
| deepseek-v4-pro + seeds + guidelines | 0.935 |
| flash + guidelines v2 (added rules from half-A errors: role requests, "ATTENTION - STOP", fake template fields) | A 1.00 / **B 0.96** -> not adopted (no gain on held-out half) |
v1 labels not reused (I relabeled the 326 pool texts with the better prompt for $0.009).

## Data and training (gte-modernbert, full FT, max_len 256)
- pool only (286 train): 0.935; + 1,148 seed-steered synthetic EN/DE inputs (6 sampling seeds x 200, kept only those the labeler
  confirmed): 0.945 / 0.930 (seeds 0/1); synthetic only: 0.840; UltraBERT instead of gte: 0.935; + 20 seeds, 8 epochs: 0.950.
- The student's best threshold was always very low (0.07-0.2): the labeler under-calls injections relative to the dataset's
  conventions (role-play requests and interrupt markers count as injections). Fix: train on the dev gold too.
- 5-fold CV over dev (each fold's model never sees its fold): dev in training x0 -> oof 0.935, x1 -> 0.935, **x3 -> 0.965** (torch),
  0.950 int8, threshold flat 0.3-0.5 -> 0.5.
- Final model = same recipe on all 200 dev x3 + pool + synth + seeds. int8 vs torch agree on 99.1% of test inputs.
- Library note: reusing one `OpenRouterClient` across two `asyncio.run` calls silently turns calls into "transport" label failures;
  `synthesize` with identical prompts is fully cached, so more data needs a different `extra_body={"seed": i}` per round.

## Files
`experiments/prompt_injections/`: i1_devlabel (+i1_devlabel_guide), i2_data, i3_train(_lib), i4_cv, i5_final.
Final model: `data/v2/prompt_injections/models/final_gte_e8_dup3/` (`model.int8.onnx`).

## Caveat / next
116 test examples: CI is wide (+-3.5 pts) but the margin over Jev is 21 pts. Next: selective fp32 layers to recover the int8 loss
(0.965 -> 0.950 on CV), more targeted synthetic data for role-play injections.
