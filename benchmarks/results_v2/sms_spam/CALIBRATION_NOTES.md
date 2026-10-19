# SMS Spam: post-hoc calibration (head `answer`, noul)

## Dev predictions (out-of-sample) and verification
- Deployed model: 5 cross-fit int8 ONNX fold models (`data/v2/sms_spam/models/gte_full_e3/fold{0-4}/model.int8.onnx`). Each test
  text is scored by the one fold model that did not train on it.
- Dev (200 gold) was carved out of the eval set before cross-fitting. Checks: 0 dev uids in `labels.jsonl`, 0 in `eval_inputs`,
  so no fold model trained on a dev example. Every fold model's dev prediction is therefore out-of-sample.
  Caveat: SMS Spam contains exact-duplicate texts. 27/200 dev texts have an exact duplicate in the eval set (about 24 per fold
  model's training data). The matching rate on test is 576/5354 (10.8%) of test texts duplicated in their scoring model's training data. The rates are similar, so dev
  reflects test in this respect.
- Inference path: `experiments/sms_spam/common.onnx_probs` (8 threads, length-sorted batches of 32, max_len 128), the same function
  `s3_onnx.py` used for the frozen test probabilities. Script: `experiments/sms_spam/calibration/c1_dev_probs.py`.
  - Dev re-prediction equals the stored `dev_probs_int8.npy` (max |diff| 0).
  - Re-predicting each full held-out fold equals the frozen `eval_probs.npy` (max |diff| 0).
  - Re-predicting a 64-text subset differs by up to 0.13. Dynamic int8 quantisation computes activation scales per batch, so a
    text's probability depends on which texts share its batch. Dev was predicted with the same batching scheme as test, so it is
    consistent. Batch-1 deployment would give slightly different probabilities (not studied here).

## Scheme: pooled over the 5 fold models
The test probabilities are a mixture: each fold model scores about 1/5 of test, and one calibrator is applied to all of them.
Averaging the 5 models' dev predictions would calibrate a 5-model ensemble that is never deployed. I stacked all 5 fold models'
dev predictions instead (5 x 200 = 1000 rows, `dev_probs_deployed.npy`, shape (5, 200)). This gives the same equal mixture of
models as test and keeps every out-of-sample prediction. The CV is grouped by dev text, so the 5 rows of one text never land on
both sides of a split. As a sensitivity check, I also ran a deterministic assignment: a seeded balanced permutation, 40 dev texts
per fold model, 200 rows (`dev_probs_deployed_assigned.npy`). Script: `experiments/sms_spam/calibration/c2_select.py`.

## Dev CV table (5-fold, out-of-fold)
| method | pooled NLL | pooled ECE | pooled ACE | pooled Brier | assigned NLL | assigned ECE | assigned Brier |
|---|---|---|---|---|---|---|---|
| none | **0.0381** | 0.0095 | 0.0085 | 0.0077 | **0.0415** | 0.0125 | 0.0069 |
| temperature | 0.0789 | 0.0046 | 0.0121 | 0.0082 | 0.0788 | 0.0041 | 0.0060 |
| platt | 0.0851 | 0.0051 | 0.0110 | 0.0100 | 0.0797 | 0.0048 | 0.0063 |
| isotonic | 0.1078 | 0.0071 | 0.0095 | 0.0094 | 0.0770 | 0.0024 | 0.0058 |

**Chosen: `none`** (lowest CV NLL under both schemes). Fitted on all dev, temperature would be T = 0.81 (sharpen) and Platt
a = 1.36, b = 0.69. On held-out folds these sharpen a few confident errors (8 pooled dev errors, 5 of them at confidence >= 0.83),
which doubles the NLL. With 24 positives in 200 dev texts there is too little signal to beat the raw probabilities.

## Test (scored once, `calibration.json`; N = 5354)
| system | ECE | ECE15 | adaptive ECE | Brier | NLL | mean conf | acc |
|---|---|---|---|---|---|---|---|
| ours raw | 0.0090 | 0.0094 | 0.0087 | 0.0092 | 0.0408 | 0.980 | 0.989 |
| ours calibrated (`none`) | 0.0090 | 0.0094 | 0.0087 | 0.0092 | 0.0408 | 0.980 | 0.989 |
| ours oracle recalibrated | 0.0014 | 0.0019 | 0.0024 | 0.0091 | 0.0369 | 0.990 | 0.989 |
| Jev raw | 0.0311 | 0.0311 | 0.0311 | 0.0146 | 0.0721 | 0.952 | 0.983 |
| Jev oracle recalibrated | 0.0015 | 0.0035 | 0.0032 | 0.0133 | 0.0516 | 0.984 | 0.983 |

Our raw model is already better calibrated than Jev on every metric: ECE 0.009 vs 0.031, Brier 0.0092 vs 0.0146, NLL 0.041 vs 0.072.
Even after the test-label oracle recalibration, Jev stays worse on Brier and NLL (0.0133 / 0.052) than our raw model.
Our model is slightly underconfident on test (conf 0.980 vs acc 0.989). The oracle's gain from fixing that is small (NLL 0.041 -> 0.037).

## Caveats
- With 200 dev texts and 24 positives, method selection is noisy. The oracle shows a small gain was available on test, but dev
  could not detect it.
- Decisions, thresholds and test probabilities are unchanged (threshold 0.5 from FROZEN.json).
- `calibration_eval` was run once; no rerun.

## Batch-1 serving
The model is served one request at a time, so I recomputed the int8 predictions at batch size 1. Everything else is the same:
the 5 fold artifacts, preprocessing, max_len 128 and 8 threads, with each test text scored by its own fold model (`te_idx`).
Script: `experiments/sms_spam/calibration/c3_batch1.py`. Outputs:
- test: `data/v2/sms_spam/eval_probs_b1.npy`
- dev, all 5 fold models: `dev_probs_b1.npy`, shape (5, 200)

Batch-1 inference is deterministic (a repeat of 20 texts gave max |diff| 0).
- Probability shift vs the frozen batched test probabilities: max |dp| 0.414, mean |dp| 0.0071.
- On dev: max 0.299, 1 flip in the 1000 pooled rows. Per-fold dev F1 at 0.5 is 0.979 / 0.979 / 0.957 / 0.957 / 0.979,
  the same as the batched int8 values in FROZEN.json.
- **Decision flips on test: 15 of 5354 (0.28%)** at the frozen threshold 0.5.
- **F1 at batch 1: 0.9573 (95% CI 0.9456-0.9678), vs 0.9580 batched.** Jev scores 0.9366 on the same examples.
  **Verdict at batch 1: still a win**, since the lower CI bound is above Jev.

Calibrator refit on the batch-1 dev predictions (`c4_batch1_select.py`), using the same pooled scheme (1000 rows, 5-fold CV grouped
by dev text, lowest NLL):

| method | CV NLL | ECE | ACE | Brier |
|---|---|---|---|---|
| none | **0.0368** | 0.0058 | 0.0076 | 0.0069 |
| temperature | 0.1044 | 0.0039 | 0.0143 | 0.0076 |
| platt | 0.0794 | 0.0051 | 0.0097 | 0.0087 |
| isotonic | 0.1318 | 0.0065 | 0.0149 | 0.0099 |

Chosen: `none`. Test calibration at batch 1 (raw = calibrated): ECE 0.0063, ECE15 0.0066, adaptive ECE 0.0049, Brier 0.0091,
NLL 0.0390, mean confidence 0.984, accuracy 0.9886. This is slightly better than batched (ECE 0.0090, NLL 0.0408) and still well
ahead of Jev raw (ECE 0.031, Brier 0.0146, NLL 0.072). `batch1_eval` was run once (`batch1.json`, logged).
