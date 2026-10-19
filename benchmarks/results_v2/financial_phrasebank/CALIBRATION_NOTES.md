# Financial PhraseBank: post-hoc calibration

## Dev predictions (out-of-sample)
- Deployed model `data/v2/financial_phrasebank/models/v1seeded_s1/model.int8.onnx` (dynamic per-channel int8), run with
  the inference path that produced the frozen test probabilities (`experiments/ag_news/finalize_probs.py`:
  `OnnxPredictor(model_dir, threads=8)`, batches of 64, max_len 96). Script: `experiments/financial_phrasebank/calibration/c1_dev_probs.py`.
- **The model did not train on dev.** `train_student.py` and `v1_silver` drop all seed and dev uids. No dev text appears verbatim in the
  3,637-text training pool (normalised exact-match check: 0 overlaps). Early stopping and temperature used held-out
  *silver*, not dev.
- Verification: the re-predicted dev probs match `dev_probs_deployed.npy` and `models/v1seeded_s1/dev_probs_int8.npy` with
  a max |diff| of 0.0. The re-predicted first 256 test inputs (`protocol.eval_inputs`, no labels) match the frozen
  `eval_probs.npy` with a max |diff| of 0.0. Dev accuracy is 0.850 (as frozen). The fp32 probabilities were not used, because int8 flips about 15% of dev argmaxes.
- File used: `benchmarks/results_v2/financial_phrasebank/dev_probs_deployed.npy` (identical to the re-prediction).

## Dev CV table (5-fold stratified, 200 gold dev; class counts neg/neu/pos = 26/122/52)
| method | NLL | ECE | ACE | Brier | mean conf | acc |
|---|---|---|---|---|---|---|
| none | 0.4138 | 0.0399 | 0.0464 | 0.2395 | 0.850 | 0.850 |
| temperature | 0.4167 | 0.0449 | 0.0490 | 0.2407 | 0.844 | 0.850 |
| vector | 0.4094 | 0.0497 | 0.0558 | 0.2307 | 0.851 | 0.845 |
| isotonic | 0.6502 | 0.0701 | 0.0636 | 0.2372 | 0.866 | 0.860 |

Fold-seed check (CV NLL, seeds 1-3): vector 0.409/0.410/0.403, none 0.414, temperature 0.418-0.420, isotonic 0.55-0.61.

## Chosen method: `none` (no parameters)
The unrestricted `select_calibrator` would pick vector scaling (T=0.889, bias neg/neu/pos = -0.074/-0.382/+0.455). I
restricted the selection to argmax-preserving methods (none, temperature), and decided this on dev before scoring, because:
- vector scaling changes the top label on 11 of 200 dev examples. The calibrated probabilities would then contradict the frozen
  deployed decisions, which the brief says must not change. Its CV accuracy also drops to 0.845.
- its CV NLL gain over `none` is 0.004, which is within the fold-seed noise. Its CV ECE is worse (0.050 vs 0.040).

Among the argmax-preserving methods, `none` has the lowest CV NLL. Temperature fitted on all of dev gives T=1.03, so the int8
model is already close to calibrated on dev.

## Test result (970 examples, scored once via `calibration_eval`)
| row | ECE | ACE | Brier | NLL | mean conf | top-label acc |
|---|---|---|---|---|---|---|
| ours raw | 0.042 | 0.044 | 0.285 | 0.476 | 0.836 | 0.794 |
| ours calibrated (none) | 0.042 | 0.044 | 0.285 | 0.476 | 0.836 | 0.794 |
| Jev raw | 0.137 | 0.136 | 0.383 | 0.703 | 0.867 | 0.731 |
| ours oracle-recalibrated | 0.035 | 0.035 | 0.260 | 0.432 | 0.829 | 0.811 |
| Jev oracle-recalibrated | 0.013 | 0.019 | 0.230 | 0.392 | 0.833 | 0.837 |

Ours as deployed beats Jev as delivered on every calibration metric: ECE is about 3x lower, NLL is 0.48 vs 0.70, and Brier is 0.29 vs 0.38.

## Caveats
- The oracle rows use test-label CV with unrestricted methods (vector/isotonic). These methods change the top label, so their accuracy differs
  (Jev oracle 0.837, ours 0.811). They are not decision-preserving and are not comparable with the deployed decisions. Jev's
  oracle row being better shows that Jev's probabilities carry more *ranking* information once they are re-mapped with test labels.
  Neither row is achievable without test labels.
- The test set is miscalibrated in a way that dev does not show. Accuracy drops from 0.85 on dev to 0.79 on test (NOTES.md: the test split mixes annotator-agreement
  levels), while confidence stays at about 0.84. Our test ECE (0.042) is still close to dev (0.040).
- With 200 dev examples and only 26 negatives, the CV differences between none, temperature and vector are within noise.

## Batch-1 serving (follow-up)
Dynamic int8 activation scales depend on batch composition and padding. The frozen probabilities came from batches of 64,
but the model is served one request at a time. I recomputed them with the same `model.int8.onnx`, `OnnxPredictor(threads=8)` and max_len 96, at
**one text per call** (`experiments/financial_phrasebank/calibration/c3_batch1.py`). Batch-1 inference is deterministic: a repeat run gives a max |diff| of 0.0.
- Outputs: `dev_probs_b1.npy`, `eval_probs_b1.npy`, `BATCH1.json` (written before scoring), and `batch1.json` from a single
  run of `benchmarks.v2.batch1_eval`.
- **Decision flips:** test 92/970 (9.5%), with a max |Δp| of 0.72. Dev flips are 18/200, with a max |Δp| of 0.58.
- **Headline accuracy:** test **0.818** at batch 1 (95% CI 0.792-0.842) vs **0.794** batched. On dev, batch 1 gives 0.830 vs 0.850 batched,
  so the size and even the direction of the change is noise from quantisation, not a systematic gain. The model was not
  re-selected: dev was not used to choose between batch modes, and batch 1 is reported only because it is the serving mode.
- **Verdict at batch 1 vs Jev (0.731): win.** The CI lower bound of 0.792 is above Jev, unchanged from the batched verdict.
- **Calibration refit on batch-1 dev** (5-fold CV, choice limited to none/temperature):

| method | NLL | ECE | ACE | Brier |
|---|---|---|---|---|
| none | 0.4007 | 0.0337 | 0.0387 | 0.2374 |
| temperature | 0.4032 | 0.0317 | 0.0438 | 0.2374 |
| vector (ref.) | 0.4079 | 0.0415 | 0.0623 | 0.2399 |
| isotonic (ref.) | 0.6563 | 0.0562 | 0.0683 | 0.2526 |

  Chosen: **none**.
- **Test calibration at batch 1** (raw = calibrated): ECE 0.054, ACE 0.054, Brier 0.274, NLL 0.457, mean confidence 0.848, accuracy 0.818.
  Jev raw on the same examples: ECE 0.137, Brier 0.383, NLL 0.703. Ours stays far better calibrated than Jev. ECE rises slightly vs batched (0.042), while
  Brier and NLL improve because accuracy is higher.
- Caveat: a 9.5% decision flip rate from batching alone confirms that this int8 model is unstable on borderline neutral/positive
  cases (see NOTES.md "Next": quantisation-aware fine-tuning, or picking the seed whose int8 agrees best with fp32).
