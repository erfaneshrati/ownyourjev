# clinc150: post-hoc calibration notes

Head `answer`, 151 options (150 intents + `oos`). Deployed model: gte-modernbert, int8 on `mlp.Wi` only
(`data/v2/clinc150/models/E_lr5_q_only_Wi/model.int8.onnx`). The deployed decision rule (predict oos when the best
in-scope probability < 0.4) is unchanged.

**Important for reading the ECE:** the frozen test vector (`eval_probs.npy`) is the probability vector *after*
`apply_threshold`. On rows where the rule fires, oos gets the in-scope max + 1e-3, and the vector is renormalised.
So argmax equals the deployed decision. The ECE/ACE here are computed on the top probability of that vector (its
argmax), but on oos-rule rows that top probability is a near-tie value of about 0.28, not a real confidence.

## Dev predictions (out-of-sample)
- The deployed model never trained on dev or seeds. Dev was used only for selection and the threshold (see NOTES.md).
  So the dev predictions come from the exact deployed artifact and path.
- That path is `experiments/clinc150/predict_eval.py`, which produced `eval_probs.npy`: `OnnxPredictor(model_dir, model.int8.onnx, threads=8)`,
  batches of 64, then `apply_threshold(p, 0.4)`.
- Script: `experiments/clinc150/calibration/predict_dev.py`.
- Verification: re-predicting the first 256 `eval_inputs` reproduces the frozen `eval_probs.npy` with **max abs diff 0.0**
  and full argmax agreement. The non-int8 `model.onnx` gives a diff of 0.26, which confirms that the int8 file is the deployed one.
  Dev accuracy is 0.965, matching FROZEN.json.
- Saved:
  - `dev_probs_deployed.npy` (post-threshold, the vector that was calibrated)
  - `data/v2/clinc150/dev_probs_deployed_prethreshold.npy`

## Method restriction (decided on dev)
200 dev examples cover 114 of the 151 classes, at most 8 per class, and only 3 classes have ≥ 5 examples. Per-class methods cannot be estimated:
- **Isotonic** (151 one-vs-rest fits): degenerate. CV NLL is 2.09 and CV accuracy falls to 0.66.
- **Vector scaling** (151 biases): its CV NLL is slightly lower (0.2155), but it changes the argmax (CV accuracy 0.960 vs 0.965).
  That violates "decisions must not change". The coordinator rule later confirmed that Choice heads are limited to argmax-preserving methods.

So the candidates were **none / temperature**. Their 5-fold CV (KFold, seed 0) is below; vector and isotonic are shown for reference only.

| method | CV NLL | ECE | ACE | Brier | acc | mean conf |
|---|---|---|---|---|---|---|
| **none** | **0.2230** | 0.0540 | 0.0526 | 0.0852 | 0.965 | 0.932 |
| temperature | 0.2567 | 0.0521 | 0.0469 | 0.0795 | 0.965 | 0.951 |
| vector (ref) | 0.2155 | 0.0358 | 0.0290 | 0.0782 | 0.960 | 0.959 |
| isotonic (ref) | 2.0919 | 0.0687 | 0.0708 | 0.4028 | 0.660 | 0.722 |

Temperature is unstable across CV seeds (NLL 0.257 / 0.229 / 0.221 vs none at 0.223 for seeds 0/1/2).
If refitted on all of dev, T would be 0.84, a mild sharpening. **Chosen: `none`** (no parameters).
Frozen in `CALIBRATION.json`, and `calibration_eval` was run once.

## Test result (5,500 utterances, 1,000 oos)
| system | ECE | ACE | Brier | NLL | acc (argmax) | mean conf |
|---|---|---|---|---|---|---|
| ours raw | 0.121 | 0.121 | 0.189 | 0.487 | 0.934 | 0.839 |
| ours calibrated (`none`) | 0.121 | 0.121 | 0.189 | 0.487 | 0.934 | 0.839 |
| ours oracle-recalibrated (test 5-fold) | 0.017 | 0.013 | 0.111 | 0.273 | 0.932 | 0.937 |
| Jev raw | 0.024 | 0.019 | 0.163 | 0.516 | 0.895 | 0.909 |
| Jev oracle-recalibrated | 0.047 | 0.053 | 0.138 | 0.384 | 0.918 | 0.872 |

- Ours has the better NLL than Jev raw (0.487 vs 0.516), but a worse ECE and Brier.
- With the oracle, ours is better on every metric.

## Caveats
- **Our test ECE is driven by the oos rule, not by the classifier.**
  - 17.9% of test rows are oos-rule predictions with top probability of about 0.28, and they are mostly correct, so the model looks heavily underconfident (mean conf 0.84 vs accuracy 0.93).
  - Dev has only 8 oos (4%), so dev CV could not see this, and the dev ECE was 0.054.
  - No argmax-preserving 1-parameter calibrator fixes it, because the oos value is tied to the in-scope max.
  - A proper fix would emit a calibrated P(oos) directly, for example by putting the threshold mass into oos instead of a near-tie. That is a change to the deployed vector, and it was not in scope here.
  - The oracle row shows that a test-fitted recalibrator, which may change the argmax, reaches ECE 0.017.
- 200 dev examples over 151 classes is too few for anything beyond a 1-parameter calibrator. The CV differences between none and temperature are within seed noise.
- The test ECE is computed on the argmax of the probabilities. The deployed decision is the threshold rule. These coincide here only because the rule is baked into the vector.

## Batch-1 serving
- Recomputed with `experiments/clinc150/calibration/predict_b1.py`: the same int8 artifact with threads=8, one text per call, then the same `apply_threshold(·, 0.4)`.
  - Saved: `data/v2/clinc150/{eval,dev}_probs_b1.npy`, plus `*_prethreshold.npy`, and a copy at `dev_probs_b1.npy` here.
  - Max |Δp| vs batched: 0.282 on test and 0.066 on dev.
- **Deployed decision flips (threshold rule): 14 / 5,500 (0.25%).** 11 of them involve oos on one side.
  - The thresholded vector is the deployed decision, so these equal batch1_eval's argmax flips.
- **Accuracy: 0.9338 at batch 1 (95% CI 0.9269 to 0.9405) vs 0.9340 batched.** Dev accuracy is unchanged at 0.965.
- **Verdict at batch 1: win vs Jev 0.8949.** The CI lower bound is above Jev's score.
- **Calibration refit on batch-1 dev** (none/temperature):
  - CV NLL: none 0.2238, temperature 0.2636 (seeds 1/2: 0.2295 / 0.2227 vs none 0.2238). The choice is **none** again.
  - Test at batch 1: ECE 0.120, ACE 0.121, Brier 0.189, NLL 0.487. These are essentially identical to the batched results.
- Files: `BATCH1.json`, `batch1.json` (batch1_eval run once).
