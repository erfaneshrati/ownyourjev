# AG News: post-hoc calibration

## Dev predictions (out-of-sample)
- The deployed model (`data/v2/ag_news/models/v1plain_5k/model.int8.onnx`) never trained on dev or seeds. Its training
  data is v1 plain silver labels with the seed and dev uids removed (`v1_silver` and `train_student.py` both drop them,
  `add_seeds=false`). I also asserted that the dev uids and the training uids do not overlap.
- Dev was predicted with the exact deployed path: `OnnxPredictor(model_dir, threads=8)` in batches of 64 starting at
  index 0, the same as `experiments/ag_news/finalize_probs.py`. The result matches `dev_probs_deployed.npy` exactly.
  It is saved as `dev_probs_deployed_answer.npy`. Dev accuracy is 0.915, as in FROZEN.json.
- Verification on test inputs (no labels): 4 batches of 64, aligned as in `finalize_probs.py` (starting at indices
  0, 1920, 3840 and 5760), reproduce `eval_probs.npy` exactly (max abs diff 0.0).
  - **Batch-composition caveat:** a first check on a strided subsample, so with different batch neighbours, differed
    by up to 0.29. Dynamic int8 quantizes activations per tensor over the whole padded batch, so the probabilities
    depend on which examples share a batch.
  - Dev and test were both produced with the 64-batch layout, so the calibrator is consistent with the frozen test
    probabilities.
  - Batch-1 serving would give slightly different probabilities. That is not measured here.
- Scripts are in `experiments/ag_news/calibration/`: `verify_and_fit.py`, `verify_aligned.py` and `fit_restricted.py`.

## Dev CV table (5-fold on 200 dev, out-of-fold)
| method | NLL | ECE | ACE | Brier | mean conf | acc |
|---|---|---|---|---|---|---|
| none | 0.3589 | 0.0478 | 0.0490 | 0.1509 | 0.957 | 0.915 |
| **temperature** | **0.3154** | 0.0339 | 0.0316 | 0.1473 | 0.909 | 0.915 |
| vector (excluded) | 0.3062 | 0.0415 | 0.0299 | 0.1436 | 0.913 | 0.915 |
| isotonic (excluded) | 0.5377 | 0.0622 | 0.0594 | 0.1670 | 0.927 | 0.900 |

- Unrestricted selection would have picked vector scaling (T=1.415, biases [0.31, -0.82, -0.06, 0.57]).
- Before scoring, the coordinator set a uniform rule on API-consistency grounds: Choice heads may only use calibrators
  that keep the argmax (none or temperature). Under that rule the selection picked **temperature, T = 1.5138**, refit
  on all 200 dev examples.

## Test result (7,600 examples, scored once)
| row | ECE | ACE | Brier | NLL | mean conf | acc |
|---|---|---|---|---|---|---|
| ours raw | 0.0571 | 0.0565 | 0.1724 | 0.3892 | 0.950 | 0.893 |
| **ours calibrated (T=1.514)** | **0.0076** | **0.0071** | **0.1656** | **0.3360** | 0.900 | 0.893 |
| ours oracle (5-fold on test) | 0.0055 | 0.0071 | 0.1657 | 0.3361 | 0.897 | 0.893 |
| Jev raw | 0.0770 | 0.0767 | 0.1913 | 0.7822 | 0.962 | 0.885 |
| Jev oracle (5-fold on test) | 0.0164 | 0.0205 | 0.1800 | 0.3708 | 0.874 | 0.885 |

- Our dev-fitted temperature nearly matches the test oracle.
- It beats Jev raw on every metric, and beats Jev even after Jev gets oracle recalibration on test labels.
- Decisions and accuracy are unchanged, because temperature scaling keeps the argmax.

## Caveats
- Only 200 dev examples. The CV standard error on ECE is around 1 to 2 points.
- The probabilities depend on batch composition, because of dynamic int8 activation quantization (see above).

## Batch-1 serving
The deployed model serves one request at a time, so I recomputed everything at batch size 1.

**Setup**
- Same int8 artifact and the same `OnnxPredictor(threads=8)`, one text per call, so there is no padding.
- Inputs: `protocol.dev` and `protocol.eval_inputs`.
- Outputs: `dev_probs_b1.npy` and `eval_probs_b1.npy`.
- Script: `experiments/ag_news/calibration/batch1.py`, log in `batch1.log`.
- Batch-1 inference is deterministic: repeating 20 texts gives a max diff of 0.0.

**Decision flips vs the frozen batched predictions**
- Test: 114 of 7,600 (1.5%), with a max |dp| of 0.54.
- Dev: 2 of 200, with a max |dp| of 0.52.

**Accuracy**

| split | batch 1 | batched (frozen) |
|---|---|---|
| test | 0.8926 (95% CI 0.8857-0.9001) | 0.8930 |
| dev | 0.915 | 0.915 |

**Verdict at batch 1:** a **win** over Jev (0.885), because the CI lower bound of 0.8857 is above Jev. The margin is
very thin: the lower bound clears Jev by only 0.0007.

**Calibration refit on batch-1 dev** (none/temperature only, 5-fold CV NLL)

| method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|
| none | 0.3772 | 0.0617 | 0.0514 | 0.1496 |
| **temperature** | **0.3252** | 0.0346 | 0.0452 | 0.1490 |
| vector (reference only) | 0.3144 | 0.0482 | 0.0362 | 0.1440 |
| isotonic (reference only) | 0.4642 | 0.0565 | 0.0627 | 0.1517 |

The chosen calibrator is temperature, **T = 1.5747**.

**Test calibration at batch 1**

| row | ECE | ACE | Brier | NLL | mean conf |
|---|---|---|---|---|---|
| ours raw (b1) | 0.0603 | 0.0602 | 0.1728 | 0.3984 | 0.953 |
| ours calibrated (b1, T=1.575) | 0.0124 | 0.0114 | 0.1654 | 0.3356 | 0.900 |
| Jev raw (unchanged) | 0.0770 | 0.0767 | 0.1913 | 0.7822 | 0.962 |
| Jev oracle (unchanged) | 0.0164 | 0.0205 | 0.1800 | 0.3708 | 0.874 |

At batch 1, our calibrated model still beats Jev raw on every metric. It also beats Jev's oracle-recalibrated
numbers on every metric.

`batch1_eval` was run exactly once.
