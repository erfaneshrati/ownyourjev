# PAWS: post-hoc calibration (head `answer`, noul)

## Dev predictions
- **Out-of-sample:** yes. `experiments/paws/p2_train.py:labelset` drops every seed and dev uid from all label sets, including the
  LLM labels, v1 labels and teacher pseudo-labels. Pseudo-labels cover pool texts only, and early stopping uses held-out silver labels.
  The deployed model never saw a dev example.
- **Artifact:** `data/v2/paws/models/ub_e3_soft__all+pseudo-ensmore__s0/model.int8.onnx`. This is the partial int8 model
  (attention matmuls and `down_proj` kept fp32); its md5 matches `model.int8_noattn_nodown.onnx` from `p4_partial_q.py`.
- **Inference path:** `experiments/paws/common.onnx_probs` (`OnnxPredictor`, length-sorted batches of 32, 8 threads), as used in
  `p4_partial_q.py`. The script is `experiments/paws/calibration/c1_dev_probs.py`, which writes `dev_probs_deployed.npy`.
  - The output is bit-identical (max |diff| 0) to the p4 dev probs behind FROZEN `accuracy_int8 = 0.885`.
- **Test re-prediction check: FAILED to reach 1e-4.**
  - Re-predicting all 8,000 test inputs with the same path (`c2_verify_full_test.py`) gives:
    - mean |diff| 0.013, max 0.57, correlation 0.997;
    - 98.7% decision agreement with `data/v2/paws/eval_probs.npy`.
  - None of these paths matched either (`c3`-`c5`):
    - batch 1;
    - unsorted batches of 16, 32, 64, 128 and 256;
    - fp32 ONNX;
    - torch s0 (torch s1 is clearly a different model).
  - No script that wrote the frozen test file was saved.
  - The differences look like noise:
    - they are unrelated to position and only weakly related to length;
    - the label-free confidence distributions match (mean confidence 0.9326 vs 0.9327; share above 0.99 is 0.524 vs 0.523).
  - **Most likely cause:** dynamic int8 computes activation scales per batch, so outputs depend on how the inputs were batched.
    The exact batching used for the test set was not recorded. The model is the right one; only the batching is unknown.
- **Sensitivity check:** the same noise on dev (batch 1 vs batches of 32: mean |diff| 0.012) does not change the selection:
  - platt is chosen both times;
  - a = 0.527 vs 0.526, b = 0.831 vs 0.824.

## Dev CV (5-fold, 200 dev examples, out-of-fold)
| method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|
| none | 0.3966 | 0.0590 | 0.0502 | 0.0936 |
| temperature | 0.3297 | 0.0336 | 0.0541 | 0.0943 |
| **platt** | **0.3079** | **0.0227** | **0.0382** | **0.0892** |
| isotonic | 0.4539 | 0.0396 | 0.0534 | 0.0933 |

**Chosen method:** platt, refit on all 200 dev examples: p' = sigmoid(0.5270 · logit(p) + 0.8313).
- The model is overconfident (slope about 0.53) and under-predicts "true" (positive offset).
- All four methods were allowed, per the coordinator's rule that noul heads may use any monotone method.
- Decisions stay at the frozen threshold of 0.5 on the raw probabilities.

## Test (8,000 examples, scored once via `calibration_eval`)
| system | ECE | ACE | Brier | NLL | mean conf |
|---|---|---|---|---|---|
| ours raw | 0.0481 | 0.0481 | 0.0897 | 0.3292 | 0.933 |
| **ours calibrated (platt, dev-fit)** | **0.0160** | **0.0154** | **0.0812** | **0.2668** | 0.878 |
| ours oracle-recalibrated (test 5-fold) | 0.0055 | 0.0104 | 0.0806 | 0.2634 | 0.893 |
| Jev raw | 0.0203 | 0.0250 | 0.1087 | 0.3483 | 0.870 |
| Jev oracle-recalibrated (test 5-fold) | 0.0053 | 0.0063 | 0.1069 | 0.3383 | 0.852 |

- **Raw vs Jev:** our raw probabilities beat Jev raw on Brier and NLL but not on ECE, because they are overconfident.
- **After dev-fit Platt:** we beat Jev raw on every metric. Brier and NLL are close to our own oracle.
- **ECE floor:** Jev's oracle ECE (0.005) is about the same as ours (0.0055). The remaining ECE gap is what 200 dev examples can buy.
- **Brier/NLL advantage:** even recalibrated, Jev stays worse than ours, because its accuracy (discrimination) is lower.
- **Decisions and accuracy:** the deployed decisions use the raw probabilities and are unchanged (accuracy 0.8845).
  The `accuracy` field in the calibrated rows of `calibration.json` comes from thresholding the calibrated probabilities and is
  not a deployed decision.

## Caveats
- **Inference noise:** dev probabilities come from the deployed artifact and code path, but the frozen test probabilities could not
  be reproduced to 1e-4 (int8 batch-composition noise of about 0.013 mean |diff|). The calibrator is fitted on probabilities with
  the same noise level and is insensitive to it (see the sensitivity check above).
- **Small fitting set:** 200 dev examples (about 23 errors) set the Platt slope and offset.
- **Unchanged:** `FROZEN.json`, `final.json`, the test probabilities and the threshold were not touched.
- **Ledger:** the first ledger call (in `c8_freeze.py`) raised on an invalid outcome string after `CALIBRATION.json` was written, and
  before scoring. The ledger line was logged with `c9_ledger.py` from the frozen file, using dev values only.

## Batch-1 serving
- **Predictions:** the frozen partial-int8 model was re-run on every test and dev text at batch size 1, as it is served.
  - Same artifact (`model.int8.onnx`), same `OnnxPredictor` preprocessing (max_len 128), 8 threads, one text per call.
  - Script: `experiments/paws/calibration/b1_predict.py`.
  - Outputs: `data/v2/paws/eval_probs_b1.npy` and `dev_probs_b1.npy`.
  - Batch-1 outputs are deterministic: rerunning gives identical values, and dev matches the earlier batch-1 probe exactly.
- **Difference from the frozen batched test probs:** mean |diff| 0.013, max 0.625.
  - **Decision flips: 1.29% (103 of 8,000).**
- **Accuracy:** **0.8811 at batch 1** (95% CI 0.874-0.889), vs 0.8845 batched (frozen). Jev scores 0.8499 on the same examples.
  - **Verdict at batch 1: still a win**, since the CI lower bound of 0.874 is above Jev.
  - Dev accuracy is 0.900 at batch 1 vs 0.885 batched.
- **Calibrator refit on the batch-1 dev predictions:** 5-fold CV NLL on dev, all binary methods allowed.

  | method | NLL | ECE | ACE | Brier |
  |---|---|---|---|---|
  | none | 0.3936 | 0.0662 | 0.0666 | 0.0919 |
  | temperature | 0.3264 | 0.0528 | 0.0667 | 0.0929 |
  | **platt** | **0.3033** | **0.0197** | **0.0376** | **0.0874** |
  | isotonic | 0.4436 | 0.0416 | 0.0526 | 0.0879 |

  - Platt is chosen again: p' = sigmoid(0.5261 · logit(p) + 0.8242), almost identical to the batched fit (0.5270 / 0.8313).
- **Test calibration at batch 1** (`batch1.json`, `batch1_eval` run once):

  | system | ECE | ACE | Brier | NLL |
  |---|---|---|---|---|
  | ours raw, batch 1 | 0.0512 | 0.0512 | 0.0905 | 0.3326 |
  | **ours calibrated, batch 1** | **0.0154** | **0.0172** | **0.0818** | **0.2686** |
  | Jev raw (unchanged) | 0.0203 | 0.0250 | 0.1087 | 0.3483 |

  - The calibrated batch-1 model beats Jev raw on every metric, as the batched one did.
- **Note:** the int8 noise explains why the test re-prediction in the first section could not match the frozen file to 1e-4.
  The batch-1 numbers are the ones that describe the served model.
