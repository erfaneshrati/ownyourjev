# SMS Spam (v2 re-certification)

**Result (test, scored once):** F1 **0.958** (95% CI 0.946-0.969) vs Jev **0.937** on the same 5,354 examples -> beats Jev.
CPU latency (ONNX int8, 4 threads, batch 1): p50 17.6 ms, **p95 31.7 ms** (gate 70 ms; machine shared, load ~7).
Label spend for this task: **$0.0007** (20 new LLM calls); the rest are reused v1 labels.

## Data / labels
- Cross-fit task: training texts = eval_inputs (5,354). Reused the v1 *seeded* labels (deepseek-v4.1-flash with 20 v1 seed
  examples; the v1 seeds were drawn from what is now the eval set, they are different from the v2 seeds) for 5,334 texts, after
  dropping v2 seeds/dev uids. The 20 eval texts without a v1 label (they were v1's own seed examples) were labeled fresh with
  the v2 seeds in the prompt (`experiments/sms_spam/s1_labels.py`). No v1 `gold` field used.
- Labeler agreement with dev gold (v1 labels exist for all 200 dev texts): plain acc 0.980 / F1 0.909; seeded acc 0.990 / F1 0.957.
  Seeded chosen. Plain vs seeded disagree on 108 of 5,334 pool texts. No new labeler/prompt was needed.

## Model (chosen on dev)
- gte-modernbert-base, full fine-tune, lr 3e-5, 3 epochs, batch 32, max_len 128 (dev p99 = 78 tokens), early stopping and
  temperature on a 10% held-out *silver* split. 5-fold cross-fit with `protocol.eval_folds` (each model scores only its held-out fold).
- Fold 0 comparison: lr 3e-5 dev F1 0.979 vs lr 5e-5 0.957 (1 example; kept the default 3e-5).
- Per-fold dev F1 @0.5, torch: 0.979 0.979 0.957 0.979 0.979. After ONNX int8: 0.979 0.958 0.957 0.957 0.979 (mean 0.966).
  The student beats its labeler on dev (0.966 vs 0.957).
- Threshold: mean dev F1 over the 5 int8 models is flat (0.966) for 0.40-0.55; 0.60 gives 0.970 (one example) and was
  rejected as noise; frozen at 0.5.
- Eval probabilities come from the int8 ONNX models (exactly what is deployed).

## Files
`experiments/sms_spam/{common,s1_labels,s2_train,s3_onnx,s4_thr}.py`; models `data/v2/sms_spam/models/gte_full_e3/fold{0-4}`
(`model.int8.onnx`); latency measured on fold0.

## Ledger
`benchmarks/results_v2/sms_spam/experiments.jsonl` (6 lines, backfilled).

## Library notes
- `asyncio.run` twice with one `OpenRouterClient` breaks the httpx client (second loop -> transport errors marked as label
  failures); create one client per `asyncio.run` (found in the prompt_injections run).

## Next
Nothing needed; possible: average the 5 fold models' dev predictions for threshold choice (done), or relabel the 108 plain/seeded
disagreements with a second labeler.
