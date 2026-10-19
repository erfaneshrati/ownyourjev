# PAWS (v2 re-certification)

**Result (test, scored once, round 1):** accuracy **0.8845** (95% CI 0.877-0.891) vs Jev **0.8499** on the same 8,000 examples -> beats Jev.
CPU latency (ONNX partial int8, 4 threads, batch 1): p50 40 ms, **p95 54.4 ms** at load avg 12.6 (shared machine; other runs
47.8 ms at load 18.6, 77.8 ms at load ~30; 46 ms at load ~7 for the same architecture). Gate 70 ms.
Label spend: **$0.154** (dev prompt comparisons ~$0.04, 3,200 new pool labels $0.11). Ledger: `experiments.jsonl` (13 lines).

## Labels (chosen on dev, 200 gold; halves A/B)
| labeler / prompt | dev acc (A / B) |
|---|---|
| deepseek-v4.1-flash plain | 0.830 (0.82/0.84) |
| flash + 20 v2 seeds | **0.875** (0.85/0.90) |
| flash + seeds + hand-written "swap" guidelines | 0.850 |
| deepseek-v4-pro + seeds | 0.835 |
Reused the v1 *seeded* labels (flash with 20 different v1 seed examples drawn from the train pool) for 4,464 pool pairs (after dropping
seeds/dev uids; 500 used as silver val), and labeled 3,200 more pool pairs with flash + v2 seeds -> 7.7k LLM labels.

## Models (dev accuracy)
- gte-modernbert on 4.5k labels: 0.740 (agreement-filtered 0.705; 8 epochs 0.735). gte does not learn PAWS from few labels.
- UltraBERT teacher on 4.5k labels: 0.870 / 0.840 (seeds 0/1). On 7.7k: 0.865 / 0.860; 2-teacher average 0.875.
- **Distillation** (the key lever): the teachers pseudo-label the other ~41k unlabeled pool pairs (soft P(true)); student trained on
  LLM labels + soft pseudo labels (48.7k pairs, `soft_labels=True`).
  - teacher s0 (4.5k) pseudo -> UltraBERT student 0.870 (2 seeds), gte student 0.860 (int8 0.845).
  - 2-teacher average of the 4.5k teachers (incl. the weak s1) -> 0.830 (failure).
  - 2-teacher average of the 7.7k teachers + all 7.7k LLM labels -> UltraBERT student **0.890 (seed 0) / 0.890 (seed 1)**. Frozen: seed 0.
- Caveat: ~10 recipes were compared on the same 200 dev examples, so the chosen dev number is mildly optimistic (SE ~0.022).

## Quantisation / latency
- Full dynamic int8 breaks UltraBERT (0.870 -> 0.625); excluding the attention act-x-act MatMuls does not help (0.625);
  **excluding attention matmuls and every `mlp.down_proj`** (73 of 217 MatMuls stay fp32) restores it: 0.865 (teacher), 0.885 (final).
  `down_proj` inputs (post-gated-activation outliers) are the int8-sensitive layer. fp32 ONNX p95 88 ms vs partial int8 ~46-54 ms.
  Implemented in `experiments/paws/p4_partial_q.py` (`quantize_dynamic(..., nodes_to_exclude=...)`); worth merging into
  `ownyourjev.export.quantize_int8` as an option.
- Eval probabilities are from the deployed partial-int8 ONNX model.

## Files
`experiments/paws/`: p1_devlabel (labeler comparison), p2_train (all students/teachers), p3_onnx, p4_partial_q, p5_pseudo, p6_ens_pseudo,
p7_label_more. Final model: `data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0/` (`model.int8.onnx` = partial int8).

## Next
More LLM labels for stronger teachers (learning curve still rising), 3+ teacher ensembles, a second distillation round, try
gte student with selective fp32 layers.
