# banking77 v2: notes

## Result (test scored once, no reruns)
- **Test accuracy 0.861 (95% CI 0.849–0.873) vs Jev-1.13 0.797 on the same 3,076 examples. Beats Jev.**
- CPU latency (onnxruntime int8, 4 threads, batch 1, measured on a busy shared machine): p50 14.6 ms, **p95 24.5 ms** (gate is 70 ms).
- Label spend (scope banking77): **$0.916 total**. $0.49 of that went into the labels the final model uses; the rest was dev experiments and one clean-estimate pool relabel.

## Final recipe
1. **Labeling prompt** (`experiments/banking77/lab.py`, `intents.py`). The trained question is unchanged from tasks.py. Only the labeler prompt is richer:
   - a hand-written description for each of the 77 intents;
   - decision rules for confusable families: card_arrival vs card_delivery_estimate; top_up failed / reverted / pending; fees by transaction type; unrecognised transactions by type; wrong exchange rate on a purchase vs an ATM withdrawal; incoming money missing; spare card vs physical card; country_support; declined vs acceptance; expiring card;
   - all 220 gold examples (20 seeds + 200 dev) as labeled examples.
2. **Batched labeling**: 50 queries per call, answered as a JSON map from number to intent. The long system prompt is paid once per batch and served from the provider prompt cache, so the full pool costs about $0.20 with deepseek-v4.1-flash. This is a new helper in the experiment folder (`lab.label_texts`) that could be merged into the library.
3. **Full pool (9,773 texts)** labeled by deepseek-v4.1-flash and qwen3.7-flash. On the 840 texts where they disagree, gemini-3.1-flash-lite and deepseek-v4-pro also label. Soft target = vote distribution over the labelers (one-hot where flash and qwen agree). Hard label = majority, with ties broken in the order pro > flash > gemini > qwen (`combine.py`).
4. **Student**: gte-modernbert, full fine-tune with soft labels (`soft_labels=True`), max_len 64, lr 3e-5, batch 32, at most 5 epochs, patience 3. Best epoch by dev loss = 3. Trained on the pool plus the 20 seeds, with dev as the validation set.
5. **Export**: ONNX with dynamic int8 quantisation (per-channel). Dev accuracy 0.905 in fp32, 0.895 in int8.

## Dev evidence (200 gold; halves A/B = even/odd indices)
Labeler accuracy on dev, batch 20, prompt = descriptions v1 + 20 seeds:
| labeler | dev acc |
|---|---|
| deepseek-v4.1-flash | 0.81 |
| deepseek-v4-pro | 0.81 |
| gemini-3.1-flash-lite | 0.81 |
| claude-haiku-5.5 | 0.805 |
| qwen3.7-flash | 0.79 |
| gemini-2.5-flash-lite | 0.79 |
| gpt-5-mini (minimal reasoning) | 0.77 |
| gpt-5.6-luna | 0.765 (20 missing) |

- For comparison, v1 silver labels (one query per call, no descriptions) agreed with gold on 0.735 of the 102 dev examples that overlap the v1 pool. **Rich descriptions plus rules give about +7 points.**
- **More gold in the prompt helps.** 2-fold dev CV (each half labeled with seeds + the other half in the prompt): flash 0.84, gemini-3.1 0.835, pro 0.835, qwen 0.81–0.82, haiku 0.805–0.81, gemini-2.5 0.81. Adding dev half A to the prompt raised flash's accuracy on half B from 0.85 to 0.86 and haiku's from 0.82 to 0.86.
- Batch size 20 vs 50 makes no difference for flash (0.84 for both).
- **Voting did not help label accuracy.** Best 3-labeler vote 0.845; the flash+qwen → gemini/pro scheme 0.835, against 0.84 for flash alone. Where flash and qwen agree (185/200), they are right 0.87 of the time.
- Students (gte-modernbert full fine-tune; dev numbers optimistic because dev was in the labeling prompt):
  - v1 silver (seeded, 4.9k): 0.775; int8 0.745. Pipeline check only.
  - flash labels, full pool: 0.89.
  - vote hard: 0.90.
  - **vote soft: 0.905 (int8 0.895), chosen.** The three are within dev noise. I picked soft because it encodes labeler uncertainty.
- **Clean estimate**: pool relabeled by flash with only seeds + dev half A in the prompt, student selected on A, scored on the unseen half B: **B = 0.83**. That is about equal to the labeler's own accuracy, so the student did not average out label noise beyond the labeler. Also adding dev half A ×3 to the training set: B stayed at 0.83 (no gain).
- On test the model got 0.861, above the clean 0.83. The final labels had all 220 gold examples in the prompt, not 120. With n=100, B's CI is about ±7 points anyway.
- int8 per-tensor quantisation was slightly worse than per-channel (0.73 vs 0.745 on the v1-silver model), so I kept per-channel.

## What worked / didn't
- Worked:
  - intent descriptions plus decision rules written from intent names, seeds and dev-A errors;
  - putting all gold examples in the labeling prompt (dataset conventions matter most on this task);
  - batched labeling (cheap, no accuracy loss);
  - the full pool;
  - gte-modernbert + int8 (fast, about 1 point int8 loss).
- Didn't help:
  - Stronger or more expensive labelers: pro, haiku, gpt-5-mini and luna did not beat flash.
  - Majority voting and stronger-labeler adjudication of disagreements: flat on dev.
  - Adding dev gold to student training: flat on B.
- Remaining errors are mostly dataset-convention ambiguities, for example vague "payment I don't recognise" being direct_debit, and beneficiary_not_allowed vs failed_transfer.

## Next ideas
- Pick in-context gold examples per batch by retrieval (cluster pool queries, show the nearest gold).
- Student/labeler disagreement relabeling.
- Ensemble 2 int8 students (latency allows it).

## Files
- Code: `experiments/banking77/` (intents.py, lab.py, dev_label.py, dev_cv.py, pool_label.py, combine.py, train_student.py, quant.py, predict_eval.py).
- Labels: `data/v2/banking77/pool/`.
- Final model: `data/v2/banking77/models/votesoft` (model.int8.onnx).
