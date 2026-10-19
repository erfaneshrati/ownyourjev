# SST-5: post-hoc calibration notes

## Dev predictions (out-of-sample)
- The deployed model is `ub_s1` (UltraBERT-150M, partial int8). Its run summary has `gold_fold: null`. It was trained only on teacher soft labels over the pool, which excludes seeds and dev, with early stopping on 400 silver examples. It never saw a dev example, so I predicted dev with the deployed artifact directly.
- Script: `experiments/sst5/calibration/predict_dev.py`. It follows the same path as `experiments/sst5/predict_final.py`, which produced `eval_probs.npy`: `OnnxPredictor("data/v2/sst5/models/final", threads=8)` loads `model.int8.onnx`, and examples run in contiguous batches of 32.
- Output: `dev_probs_deployed.npy`. It is identical to the frozen `dev_probs.npy` (max diff 0.0). Dev accuracy is 0.600, matching FROZEN.json.
- Verification on test inputs (no labels), in `experiments/sst5/calibration/verify_eval.py` and `data/v2/sst5/calib_verify.json`:
  - I re-predicted four of the original 32-example eval batches (offsets 0, 32, 1024 and 2176). They match `eval_probs.npy` exactly (max diff 0.0).
  - A random 64-example subset, batched differently, gave a max diff of 0.17. The cause is dynamic int8: activation quantisation scales are computed per batch tensor, padding included, so a text's probabilities depend on which other texts share its batch.
  - Batch-1 inference (the latency setting) on the first 32 test texts differs from the frozen probabilities by up to 0.18, and 1 of the 32 argmaxes flips. See the caveats.

## Dev CV (5-fold stratified, seed 0; 200 examples; gold = score rounded to level)
| method | NLL | ECE | ACE | Brier | acc | mean conf |
|---|---|---|---|---|---|---|
| none | 1.0128 | 0.138 | 0.129 | 0.556 | 0.600 | 0.704 |
| **temperature** | **0.9619** | 0.049 | 0.077 | 0.539 | 0.600 | 0.604 |
| vector | 0.9647 | 0.047 | 0.071 | 0.539 | 0.615 | 0.610 |
| isotonic | 1.2202 | 0.080 | 0.117 | 0.547 | 0.585 | 0.648 |

- The choice is stable across CV seeds 1 to 4: temperature has the lowest NLL in 3 of 4. With seed 2, vector scaling is lower by 0.0005, which is noise.
- **Chosen method: temperature, T = 1.484, refit on all 200 dev examples.** It preserves the argmax, so decisions are unchanged.

## Test (2,210 examples; scored once with `calibration_eval`)
| system | ECE | ACE | Brier | NLL | acc | mean conf |
|---|---|---|---|---|---|---|
| ours raw | 0.110 | 0.116 | 0.572 | 1.012 | 0.583 | 0.692 |
| **ours calibrated (dev T)** | **0.034** | **0.041** | **0.550** | **0.969** | 0.583 | 0.590 |
| ours oracle (test 5-fold CV) | 0.035 | 0.041 | 0.549 | 0.966 | (0.577) | 0.604 |
| Jev raw | 0.170 | 0.170 | 0.591 | 1.375 | 0.579 | 0.749 |
| Jev oracle (test 5-fold CV) | 0.036 | 0.039 | 0.549 | 0.987 | (0.572) | 0.597 |

- The raw student is overconfident (confidence 0.69 against accuracy 0.58), though less so than Jev raw (0.75 against 0.58).
- Temperature fitted on 200 dev examples reaches the test oracle bound on every metric, and beats Jev on every metric, raw and oracle-recalibrated.
- The oracle rows' accuracy can move because the oracle may pick vector or isotonic scaling. Their accuracy is not a deployed result.

## Caveats
- Probabilities depend on batch composition (dynamic int8). The frozen test probabilities and the dev fit both use batch-32 inference in dataset order, so they are consistent with each other. Batch-1 serving would produce somewhat different probabilities (one argmax flip in 32 texts in the spot check). T = 1.48 should transfer roughly, but it was not fitted to batch-1 outputs.
- With 200 dev examples, the CV differences between temperature and vector scaling (0.003 NLL) are noise. Temperature was picked by the pre-set criterion.

## Batch-1 serving (follow-up)
- **Predictions.** `experiments/sst5/calibration/predict_b1.py` uses the same artifact (`model.int8.onnx`), the same `OnnxPredictor` and threads=8 as before, but runs one text per call.
  - Outputs: `eval_probs_b1.npy` (2,210 examples) and `dev_probs_b1.npy` (200 examples).
  - Batch-1 output depends only on the text: a repeat on 20 dev texts gave a max diff of 0.0.
- **Decision flips.** On test, 113 of 2,210 argmaxes change (5.1%), and the largest probability change is 0.248. On dev, 5 of 200 change, but dev accuracy stays at 0.600.
- **Headline (argmax accuracy).**
  - Batch 1: **0.580** (95% CI 0.559 to 0.600). Batched: 0.583. Jev: 0.579.
  - **Verdict at batch 1: tie**, unchanged. The margin over Jev shrinks from +0.004 to +0.001.
- **Calibration at batch 1.** Refit on batch-1 dev, with 5-fold CV over none and temperature (argmax-preserving only):
  - none: NLL 1.011, ECE 0.127. Temperature: NLL 0.961, ECE 0.047.
  - Chosen: **temperature, T = 1.481**. This is essentially the batched T (1.484).
- **Test at batch 1.**

| system | ECE | ACE | Brier | NLL | acc | mean conf |
|---|---|---|---|---|---|---|
| ours raw | 0.114 | 0.114 | 0.571 | 1.009 | 0.580 | 0.694 |
| ours calibrated | 0.034 | 0.037 | 0.549 | 0.965 | 0.580 | 0.593 |
| Jev raw (from calibration.json) | 0.170 | 0.170 | 0.591 | 1.375 | 0.579 | 0.749 |

- The calibration conclusions do not change at batch 1: calibrated ours beats Jev raw on every metric. It also matches Jev's best possible recalibration (oracle ECE 0.036, NLL 0.987).
- `batch1_eval` was run once; the call is logged in TEST_ACCESS.log.
