# Prompt Injections: calibration study (head `answer`, Noul)

## Dev predictions (out-of-sample)
The deployed model (`final_gte_e8_dup3`) trained on all 200 dev examples x3 + seeds, so its dev predictions are in-sample and
were not used. Instead I used the saved out-of-fold predictions of the original 5-fold CV over dev with the identical recipe
(`i4_cv.py gte_e8 3`: pool + synth + seeds + the other 4 dev folds x3, held-out fold never trained on), each fold model exported to
dynamic int8 ONNX exactly like the deployed one (`i5_final.py`): `data/v2/prompt_injections/oof_cv_gte_e8_dup3_int8.npy`,
copied to `dev_probs_deployed_answer.npy`. No retraining needed.

Verification (`experiments/prompt_injections/calibration/c1_verify.py`):
- re-predicting all 200 dev examples from the 5 fold int8 ONNX files reproduces the saved OOF exactly (max |diff| 0.0); OOF acc@0.5 = 0.950;
- re-predicting all 116 test inputs with the deployed int8 file via the same `onnx_probs` path reproduces `eval_probs.npy` exactly (0.0).
  Note: dynamic int8 quantizes activations per batch, so probabilities depend on batch composition (batch=1 or a different subset
  differs by up to 0.018). The OOF dev probs were produced with the same length-sorted batch-32 path, so they match the deployed runtime.

## Dev CV table (5-fold CV of the calibrator over the 200 OOF probs; `select_calibrator`, criterion NLL)
| method | NLL | ECE | ACE | Brier | acc |
|---|---|---|---|---|---|
| **none** | **0.136** | 0.017 | 0.022 | 0.034 | 0.950 |
| temperature | 0.152 | 0.009 | 0.026 | 0.035 | 0.950 |
| platt | 0.156 | 0.028 | 0.027 | 0.036 | 0.950 |
| isotonic | 0.224 | 0.028 | 0.026 | 0.039 | 0.945 |

**Chosen: `none`** (no parameters). The raw OOF probabilities are already well calibrated (mean conf 0.957 vs acc 0.950);
every fitted method loses on held-out NLL with only 200 examples (10 errors).

## Test (116 examples, scored once)
| system | ECE | ACE | Brier | NLL | acc |
|---|---|---|---|---|---|
| ours raw | 0.020 | 0.014 | 0.032 | 0.112 | 0.957 |
| ours calibrated (= raw, method none) | 0.020 | 0.014 | 0.032 | 0.112 | 0.957 |
| ours oracle-recalibrated (5-fold on test) | 0.025 | 0.028 | 0.032 | 0.105 | 0.948 |
| Jev raw | 0.169 | 0.182 | 0.192 | 0.590 | 0.741 |
| Jev oracle-recalibrated (5-fold on test) | 0.067 | 0.056 | 0.064 | 0.295 | 0.914 |

Decisions are unchanged (threshold 0.5 on raw probs). The oracle rows' accuracy is the 0.5 decision of the recalibrated
probabilities, which can flip (Jev's isotonic/Platt map moves many items across 0.5); it is not a deployed accuracy.

## Caveats
- 116 test examples (~5 errors for us): ECE/NLL estimates are noisy; differences of a few hundredths among our rows are not meaningful.
  The gap to Jev raw (ECE 0.17 vs 0.02, NLL 0.59 vs 0.11) is large; even oracle-recalibrated Jev stays worse than our raw model.
- OOF fold models trained on 160 dev x3 instead of 200 x3, so they are a slightly weaker proxy for the deployed model (oof acc 0.950 vs test 0.957).
- Dev has only 10 OOF errors, so the calibrator selection itself has little power; `none` is also the conservative choice.

## Batch-1 serving (follow-up)
Dynamic int8 activation scales are computed per input tensor, so batched probabilities depend on batch mates and padding.
Recomputed one text per `OnnxPredictor.probs` call (same int8 artifacts, preprocessing, threads=8;
`experiments/prompt_injections/calibration/c3_batch1.py`):
- test: deployed `final_gte_e8_dup3/model.int8.onnx` -> `data/v2/prompt_injections/eval_probs_b1.npy`;
- dev: out-of-fold, each of the 5 CV fold int8 models on its held-out fold -> `dev_probs_b1.npy` (also copied here).

Probability shifts vs batched: test max |dp| 0.203, dev max |dp| 0.353. Dev OOF acc@0.5: 0.940 at batch 1 vs 0.950 batched.

**Test (scored once with `batch1_eval`):** 1 decision flip out of 116 (0.86%). Accuracy at batch 1 **0.966** (95% CI 0.931-0.991)
vs 0.957 batched. Jev on the same examples: 0.741, so the batch-1 verdict is still **win**.

Calibrator refit on the batch-1 dev OOF probs (5-fold CV, NLL):
| method | NLL | ECE | ACE | Brier | acc |
|---|---|---|---|---|---|
| none | 0.156 | 0.028 | 0.034 | 0.040 | 0.940 |
| temperature | 0.155 | 0.025 | 0.023 | 0.039 | 0.940 |
| **platt** | **0.153** | 0.017 | 0.020 | 0.038 | 0.950 |
| isotonic | 0.226 | 0.025 | 0.023 | 0.038 | 0.955 |

Chosen: **platt** (a = 0.750, b = 0.513). The margin over none and temperature is small (0.003 NLL) and within noise.
Decisions still use the frozen 0.5 threshold on the raw probabilities.

Test calibration at batch 1:
| | ECE | ACE | Brier | NLL | mean conf |
|---|---|---|---|---|---|
| ours raw (b1) | 0.023 | 0.030 | 0.032 | 0.122 | 0.965 |
| ours calibrated (b1, platt) | 0.022 | 0.039 | 0.026 | 0.098 | 0.952 |
| Jev raw (from calibration.json) | 0.169 | 0.182 | 0.192 | 0.590 | 0.910 |

Caveat: 116 examples. A single flip moves accuracy by 0.009. The batched vs batch-1 difference is within noise.
