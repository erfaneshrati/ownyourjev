# Emotion: post-hoc calibration (head `answer`, 6 classes)

## Dev predictions
- Source: `dev_probs_deployed.npy`, written by `experiments/ag_news/finalize_probs.py` in the same run as the frozen
  `eval_probs.npy`. Both used the deployed artifact: fp32 ONNX `data/v2/emotion/models/g1_10k_fp32/model.onnx`, `OnnxPredictor`, 8 threads, batch 64, `e.state` inputs.
- Out-of-sample: the student trained only on 10k LLM-labeled pool texts. It never trained on dev texts or labels. Dev half A
  shaped the labeling guidance, so it affected the training labels but not the training texts.
- Verification (`experiments/emotion/calibration/verify_dev.py`): re-predicting 128 dev and 128 test inputs gave max abs diff
  **0.0** against both saved files. Dev accuracy is 0.800.

## Dev CV (5-fold stratified, seed 0, 200 dev; class counts sadness 57, joy 78, love 16, anger 23, fear 24, surprise 2)
| method | NLL | ECE | ACE | Brier | OOF acc |
|---|---|---|---|---|---|
| none | 0.734 | 0.134 | 0.126 | 0.325 | 0.800 |
| **temperature** | **0.615** | 0.043 | 0.051 | 0.301 | 0.800 |
| vector (not eligible) | 0.561 | 0.031 | 0.043 | 0.277 | 0.810 |
| isotonic OvR (not eligible) | 0.890 | 0.067 | 0.073 | 0.286 | 0.820 |

## Method: temperature, T = 1.640 (fit on all 200 dev)
I limited the candidates to argmax-preserving methods (none/temperature) and made that choice on dev, as banking77 did. The reasons:
- The brief says decisions must not change.
- Vector scaling moved 4% of dev argmaxes, so its top label would disagree with the deployed decision.
- Vector scaling also fits a per-class bias for `surprise` from only 2 dev examples.

Vector had the lowest NLL across 5 CV seeds (0.561-0.595 vs temperature 0.615-0.621), so this choice gives up some dev NLL to keep
decisions consistent. The first ledger line (vector, from the unrestricted run) is superseded by the second. The script is `experiments/emotion/calibration/fit.py`.

## Test (2,000 examples, scored once with `calibration_eval`)
| system | ECE | ACE | Brier | NLL | mean conf | acc |
|---|---|---|---|---|---|---|
| ours raw | 0.103 | 0.101 | 0.310 | 0.757 | 0.912 | 0.811 |
| **ours calibrated (T=1.64)** | **0.026** | **0.026** | **0.295** | **0.636** | 0.810 | 0.811 |
| ours oracle-recalibrated | 0.015 | 0.016 | 0.278 | 0.598 | 0.819 | 0.819 |
| Jev raw | 0.279 | 0.279 | 0.672 | 2.831 | 0.864 | 0.585 |
| Jev oracle-recalibrated | 0.012 | 0.019 | 0.548 | 1.135 | 0.598 | 0.596 |

## Caveats
- The oracle rows are fitted on test labels by CV. They are upper bounds, not deployable results. Oracle selection can include vector/isotonic,
  which change argmax, so oracle accuracy differs from the raw accuracy.
- Jev's raw ECE is 0.28 because it is overconfident (conf 0.86 vs acc 0.585). Recalibrated with test labels, Jev's ECE matches ours, but its
  Brier and NLL stay much worse because those reflect accuracy and discrimination.
- With 200 dev examples, `surprise` has 2 instances, so per-class calibration is poorly estimated. The single-parameter temperature avoids this.
