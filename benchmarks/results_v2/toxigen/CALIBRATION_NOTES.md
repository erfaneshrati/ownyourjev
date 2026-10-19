# toxigen: post-hoc calibration

## Dev predictions
- The deployed model never trained on dev. `P.pool()` excludes the dev and seed uids, `train_mt.py` also drops dev and seed uids
  (`excl`) from both the LLM labels and the v1 silver labels, and dev was used only for config choice and the 0.30 threshold.
  So the dev predictions come from the **exact deployed artifact**, which is `data/v2/toxigen/models/soup_silv05_s012/model.onnx` (fp32),
  run through the same `mt_model.onnx_probs` path that `export_eval.py` used, with max_len 64.
  Script: `experiments/toxigen/calibration/predict_dev.py`.
- `toxic` = E[toxicity]/4 = probs[:,1:] @ [0, .25, .5, .75, 1]. `toxicity` = probs[:,1:].
- Verification:
  - Re-predicting 20 eval_inputs (every 47th) reproduces `eval_probs_toxic.npy` and `eval_probs_toxicity.npy` with max abs diff **0.0**.
  - Dev predictions also equal the saved `dev_probs_fp32onnx.npy` exactly.
- Saved files: `dev_probs_deployed_toxic.npy` (200,) and `dev_probs_deployed_toxicity.npy` (200, 5).
- Gold:
  - `toxic` is the binary gold (57 of 200 positive).
  - `toxicity` is the gold level index from `calibration_eval.gold_index`. Dev level counts are [89, 48, 26, 15, 22].

## Dev CV table (5-fold, `select_calibrator`, unrestricted; `experiments/toxigen/calibration/fit.py`)
| head | method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|---|
| toxic | none | 0.3427 | 0.0737 | 0.0778 | 0.1068 |
| toxic | temperature | 0.3269 | 0.0623 | 0.0686 | 0.1073 |
| toxic | **platt** | **0.2533** | 0.0328 | 0.0301 | 0.0777 |
| toxic | isotonic | 0.3721 | 0.0481 | 0.0535 | 0.0889 |
| toxicity | none | 1.0389 | 0.0870 | 0.0901 | 0.5638 |
| toxicity | temperature | 1.0399 | 0.0573 | 0.0774 | 0.5632 |
| toxicity | **vector** | **0.9556** | 0.0504 | 0.0589 | 0.5270 |
| toxicity | isotonic | 1.2507 | 0.1105 | 0.0841 | 0.5608 |

## Chosen (frozen in CALIBRATION.json, fitted on all 200 dev)
- **toxic: platt**, a = 2.782, b = 2.179, i.e. sigmoid(2.78·logit(E[tox]/4) + 2.18). It is monotone, so the ranking is unchanged.
  The deployed decision stays at raw ≥ 0.30. That cut corresponds to a calibrated probability of about 0.46.
- **toxicity: vector scaling**, T = 0.640, bias = [−1.365, −0.487, 0.083, 0.099, 1.670]. It **changes the argmax**: the large bias on
  level 4 moves mass toward "extremely offensive".

## Test result (scored once, 2026-10-19T13:43:56, CALIBRATION.json digest 30c1e167ab3a)
| head | row | ECE | ACE | Brier | NLL |
|---|---|---|---|---|---|
| toxic | ours raw | 0.034 | 0.047 | 0.154 | 0.467 |
| toxic | ours calibrated (platt) | 0.032 | 0.036 | 0.110 | 0.359 |
| toxic | Jev raw | 0.014 | 0.023 | 0.087 | 0.292 |
| toxic | ours oracle | 0.020 | 0.023 | 0.110 | 0.353 |
| toxic | Jev oracle | 0.022 | 0.030 | 0.086 | 0.284 |
| toxicity | ours raw | 0.120 | 0.118 | 0.680 | 1.316 |
| toxicity | ours calibrated (vector) | 0.059 | 0.055 | 0.603 | 1.131 |
| toxicity | Jev raw | 0.200 | 0.200 | 0.664 | 1.592 |
| toxicity | ours oracle | 0.027 | 0.048 | 0.584 | 1.090 |
| toxicity | Jev oracle | 0.029 | 0.041 | 0.581 | 1.095 |

## Caveats
- **Late rule change.** After I had scored, the coordinator issued a uniform rule: Choice and Score heads may use only
  argmax-preserving calibrators (none or temperature), decided on API-consistency grounds. Vector scaling violates it for
  `toxicity`. Per the coordinator's instruction, I did **not** re-score.
  - Under the restricted set, the dev CV picks **none**: NLL 1.0389 vs temperature 1.0399, both from the table above, so no test
    information is involved. The compliant `toxicity` result is therefore identical to "ours raw": ECE 0.120, NLL 1.316. That is
    still better calibrated than Jev raw (ECE 0.200, NLL 1.592), but its Brier is worse (0.680 vs 0.664).
  - CALIBRATION.json was left as scored, so its digest still matches the TEST_ACCESS.log entry.
- **ECE and "accuracy" for `toxic` use a 0.5 cut.** `calibration_metrics` measures top-label confidence and correctness at 0.5.
  The deployed decision is at 0.30 (deployed test accuracy 0.854), so the "ours raw" accuracy at 0.5 is 0.761. The raw score
  is not a probability. Platt rescales it, and NLL and Brier improve a lot (0.467 → 0.359, 0.154 → 0.110), reaching the
  oracle level. Jev is still better on `toxic`, which reflects its better discrimination (AUROC 0.949 vs 0.924), not miscalibration.
- n = 200 dev (57 positives) gives noisy calibrator fits. The 5-bias vector fit for `toxicity` matched the oracle trend on test,
  but that is incidental.
