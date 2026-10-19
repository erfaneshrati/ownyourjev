# toxic_chat: post-hoc calibration notes

## Dev predictions (out-of-sample)
- Deployed model: round-2 frozen `data/v2/toxic_chat/models/soupAB_sq0.75/model.int8.onnx`. This is one shared gte-modernbert encoder with two sigmoid heads, exported with eager attention, SmoothQuant alpha 0.75, then dynamic int8.
- The model never trained on dev or on the seeds. `train_student.py` blocks dev and seed uids from the training labels, and early stopping used a silver pool split. Dev was used only for selection (soup choice and thresholds), not as training data. So it is valid to predict dev with the deployed artifact directly.
- Script: `experiments/toxic_chat/calibration/predict_dev.py`. It uses the same path as `predict_eval.py`, which produced the frozen test probabilities: `mh.onnx_probs`, batch 1, max_len 512, truncation, 8 threads.
- Verification: re-predicting the first 60 `eval_inputs` reproduced the frozen test probabilities exactly (max |dp| = 0 on both heads). The new dev predictions are also identical to the existing `runs/soupAB_sq0.75/dev_probs_int8.npy`.
- Saved as `dev_probs_deployed_toxic.npy` and `dev_probs_deployed_jailbreak.npy`. Dev has 200 examples, with 16 toxic and 7 jailbreak positives.

## Dev CV (5-fold stratified, seed 0, out-of-fold metrics; `experiments/toxic_chat/calibration/fit.py`)
| head | method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|---|
| toxic | none | 0.1257 | 0.0754 | 0.0538 | 0.0273 |
| toxic | **temperature** | **0.0875** | 0.0180 | 0.0234 | 0.0247 |
| toxic | platt | 0.0882 | 0.0162 | 0.0227 | 0.0249 |
| toxic | isotonic | 0.1848 | 0.0215 | 0.0165 | 0.0248 |
| jailbreak | none | 0.1118 | 0.0736 | 0.0645 | 0.0199 |
| jailbreak | **temperature** | **0.0522** | 0.0158 | 0.0162 | 0.0154 |
| jailbreak | platt | 0.0804 | 0.0220 | 0.0186 | 0.0227 |
| jailbreak | isotonic | 0.2349 | 0.0269 | 0.0205 | 0.0246 |

- Robustness of the choice: with CV seeds 1–4, temperature was again the best by NLL for both heads in 8 of 8 cases. Platt was close on toxic for some seeds (0.090 vs 0.085) but worse on others (0.122 vs 0.090).
- Isotonic was always the worst by NLL (toxic 0.13–0.25, jailbreak 0.11–0.29), even though its binned ECE looks fine. With 16 (toxic) or 7 (jailbreak) positives, about 3 or 1–2 positives fall in each held-out fold. The step function then assigns probability near 0 or 1 to regions it has not seen, and held-out positives that land on a 0 step get very large log-loss. This is the expected small-positive-count overfit, and the CV caught it without any manual restriction. I kept all four candidates and let the CV decide.

## Chosen calibrators (refit on all 200 dev examples)
- toxic: temperature, **T = 0.515**
- jailbreak: temperature, **T = 0.435**

T < 1 means the deployed model is **underconfident**: its logits are sharpened by about 2x. This is plausible for a student trained with soft-label BCE on averaged LLM soft labels, followed by a weight soup. Both tend to pull probabilities toward the middle.

Temperature scaling is monotone, so it cannot change any ranking. Deployed decisions still use the frozen thresholds on the raw probabilities. In calibrated space those thresholds correspond to toxic 0.33 -> 0.202 and jailbreak 0.65 -> 0.806.

## Test (5,083 examples; `calibration.json`; scored once, no rerun)
| head | system | ECE | ACE | Brier | NLL |
|---|---|---|---|---|---|
| toxic | ours raw | 0.0608 | 0.0582 | 0.0260 | 0.1238 |
| toxic | **ours calibrated (T=0.515)** | **0.0040** | **0.0061** | 0.0217 | 0.0814 |
| toxic | ours oracle-recalibrated | 0.0029 | 0.0050 | 0.0217 | 0.0818 |
| toxic | Jev raw | 0.0153 | 0.0126 | 0.0199 | 0.0735 |
| toxic | Jev oracle-recalibrated | 0.0057 | 0.0080 | 0.0198 | 0.0671 |
| jailbreak | ours raw | 0.0791 | 0.0791 | 0.0128 | 0.0960 |
| jailbreak | **ours calibrated (T=0.435)** | **0.0052** | **0.0059** | **0.0048** | 0.0230 |
| jailbreak | ours oracle-recalibrated | 0.0009 | 0.0018 | 0.0041 | 0.0197 |
| jailbreak | Jev raw | 0.0308 | 0.0308 | 0.0124 | 0.0619 |
| jailbreak | Jev oracle-recalibrated | 0.0013 | 0.0004 | 0.0041 | 0.0169 |

- **toxic:** after calibration, our ECE and ACE are better than Jev's raw values and about equal to Jev's oracle values. The dev-fitted temperature nearly matches our own test oracle; on NLL it is slightly better, 0.0814 vs 0.0818. Jev still has better Brier (0.0199 vs 0.0217) and NLL (0.0735 vs 0.0814), raw and oracle. This matches toxic being a tie on F1: on this head Jev's scores separate the classes somewhat better, and calibration cannot fix discrimination.
- **jailbreak:** calibrated, we beat Jev raw on every metric: ECE 0.005 vs 0.031, Brier 0.0048 vs 0.0124, NLL 0.023 vs 0.062. With oracle recalibration for both systems, we tie on Brier (0.0041 each) and are slightly behind on NLL (0.0197 vs 0.0169).

## Caveats
- The positives are few: 16 and 7 on dev, so each dev fold holds 1–3 positives. With these counts only one- or two-parameter maps can be fitted reliably. The temperature estimate itself is noisy, but the test result shows it transferred well.
- ECE is measured on top-label confidence with a 0.5 decision. Both heads are about 97–99% negatives, so ECE is dominated by the confident-negative bin. Brier and NLL are the more informative metrics here.
- Dev was also used to select the soup and thresholds (not to train). This makes the dev predictions slightly optimistic as a selection target, but they are not in-sample for the model weights.
- Accuracy and the deployed decisions are unchanged: the thresholds and the frozen test probabilities were not touched.
