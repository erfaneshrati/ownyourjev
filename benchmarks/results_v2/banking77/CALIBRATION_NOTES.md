# banking77: post-hoc calibration

## Dev predictions
- Deployed model: `data/v2/banking77/models/votesoft/model.int8.onnx`. It was run through `OnnxPredictor(threads=4)` in batches of 64, the same path as `experiments/banking77/predict_eval.py`, which produced `eval_probs.npy`. Script: `experiments/banking77/calibration/predict_dev.py`.
- Out-of-sample check: the training set is the pool plus the 20 seeds (`train_student.build`). No dev example overlaps it, by uid or by normalised text. The dev examples were never training texts.
- Reproduction check: I re-predicted the first 128 eval inputs (no labels). The max absolute difference from the frozen test probabilities is 0.0.
- Dev int8 accuracy is 0.890 at batch 64. NOTES reports 0.895, from a different batching; int8 with padding differs slightly between batch sizes.
- Saved to `dev_probs_deployed.npy` (200×77).

## Dev 5-fold CV (seed 0, out-of-fold)
| method | NLL | ECE | ACE | Brier | mean conf | OOF acc |
|---|---|---|---|---|---|---|
| none | 0.522 | 0.105 | 0.112 | 0.209 | 0.785 | 0.890 |
| temperature | 0.451 | 0.046 | 0.049 | 0.192 | 0.895 | 0.890 |
| vector | 0.420 | 0.059 | 0.052 | 0.195 | 0.901 | 0.865 |
| isotonic (OvR) | 1.532 | 0.112 | 0.108 | 0.322 | 0.868 | 0.760 |

NLL with CV seeds 1–4:

| method | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| temperature | 0.454 | 0.454 | 0.461 | 0.454 |
| vector | 0.453 | 0.425 | 0.460 | 0.455 |

## Choice: temperature, T = 0.736
I restricted the candidates to none and temperature, and decided this on dev:
- Isotonic one-vs-rest overfits badly with 77 classes and 200 examples.
- Vector scaling (77 biases) changes the top-1 label: out-of-fold dev accuracy falls from 0.890 to 0.865. The study requires decisions to stay unchanged.
- Vector scaling is also worse than temperature on ECE and Brier. Its NLL edge appears only for CV seeds 0 and 2; on seeds 1, 3 and 4 it ties or loses.

T < 1 sharpens the probabilities. The student trained on soft vote labels, so it is underconfident: raw mean confidence is 0.785 against 0.89 accuracy on dev.

## Test (3,076 examples, scored once)
| system | ECE | ACE | Brier | NLL | mean conf | acc |
|---|---|---|---|---|---|---|
| ours raw | 0.104 | 0.104 | 0.235 | 0.644 | 0.757 | 0.861 |
| **ours calibrated (T=0.736)** | **0.024** | **0.021** | **0.218** | **0.582** | 0.873 | 0.861 |
| ours, oracle recalibrated | 0.012 | 0.016 | 0.197 | 0.493 | 0.872 | 0.872* |
| Jev raw | 0.086 | 0.086 | 0.309 | 1.234 | 0.884 | 0.797 |
| Jev, oracle recalibrated | 0.057 | 0.055 | 0.270 | 0.795 | 0.770 | 0.816* |

*Oracle rows use 5-fold CV on the test labels and may pick methods that move the argmax. They are upper bounds, not deployable.

Calibrated ours beats Jev raw on every metric: ECE 0.024 vs 0.086, Brier 0.218 vs 0.309, NLL 0.582 vs 1.234. It also beats Jev's oracle-recalibrated row.

## Caveats
- Dev chose the training epoch (early stopping on dev loss), and all 200 dev examples were in the labeling prompt. Dev is therefore optimistic: accuracy is 0.89 on dev and 0.861 on test. A temperature fitted on dev is slightly too sharp for test: calibrated mean confidence is 0.873 against 0.861 accuracy.
- Test was scored once, with no reruns.

## Batch-1 serving
The frozen `eval_probs.npy` was computed in batches of 64 (`predict_eval.py`). The deployed model serves one request at a time. With dynamic int8, a text's activation scales depend on its whole batch, padding included.

I recomputed test and dev with the same artifact, 4 threads and one text per call (`experiments/banking77/calibration/batch1.py`). Outputs: `eval_probs_b1.npy` and `dev_probs_b1.npy`.

**Prediction changes vs batched:**
- Test: max |Δp| = 0.756; **114 / 3,076 top-1 decisions flip (3.7%)**.
- Dev: max |Δp| = 0.537; 6 / 200 decisions flip. Dev accuracy is 0.895 at batch 1, against 0.890 batched.

**Headline (accuracy):**
- Batch 1: **0.871 (95% CI 0.859–0.883)**, against 0.861 batched.
- Jev on the same examples: 0.797. **Verdict at batch 1: win.** The finding is unchanged; batch 1 is about 1 point better.

**Recalibration:** I refitted on the batch-1 dev predictions with 5-fold CV NLL over none and temperature only.

| method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|
| none | 0.475 | 0.058 | 0.081 | 0.194 |
| temperature | 0.457 | 0.071 | 0.061 | 0.189 |

Temperature was chosen, with **T = 0.848**. At batch 1 the raw model is much less underconfident than when batched.

**Test calibration at batch 1** (`batch1.json`, scored once):

| system | ECE | ACE | Brier | NLL | mean conf | acc |
|---|---|---|---|---|---|---|
| ours raw, batch 1 | 0.043 | 0.045 | 0.211 | 0.564 | 0.828 | 0.871 |
| **ours calibrated, batch 1 (T=0.848)** | **0.027** | **0.024** | **0.206** | **0.548** | 0.880 | 0.871 |
| Jev raw | 0.086 | 0.086 | 0.309 | 1.234 | 0.884 | 0.797 |

**Batched vs batch 1:**
- Much of the batched model's underconfidence was a batching/padding artifact. At batch 1, raw ECE drops from 0.104 to 0.043.
- The batch-1 numbers are the ones that describe the served model.
- Calibrated at batch 1, the model beats Jev raw on ECE, Brier and NLL.
