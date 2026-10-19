# Calibration study brief (read fully before starting)

## Goal
Make the probabilities of your task's **frozen, deployed** model calibrated, and compare them with Jev's on the
same test examples. Accuracy and decisions must not change. This is a post-hoc calibration of the existing
model, not a new model.

## Rules (no cheating, same spirit as AGENT_BRIEF.md)
1. **Fit only on dev.** Calibrators are fitted on the 200 gold dev examples (`benchmarks.v2.protocol.dev(task)`)
   using the deployed model's predictions on those examples.
2. **Dev predictions must be out-of-sample.** Check `NOTES.md` / `FROZEN.json` to see whether the deployed
   model trained on any dev example.
   - If it did not, predict dev with the **exact deployed artifact and inference path** that produced the test
     probabilities in `FROZEN.json`: same ONNX/int8 file, same preprocessing, truncation, soup, ensembling and
     fold logic. Verify this by re-predicting a few test inputs (`protocol.eval_inputs`, no labels) and checking
     they match the frozen test probabilities to within ~1e-4.
   - If it did train on dev (or seeds), use out-of-fold dev predictions from models trained with the identical
     recipe minus the held-out dev fold. Reuse existing CV runs if they are saved, otherwise retrain them. If
     neither is feasible, explain why and fall back to the `none` method.
3. **Pick the method on dev only.** Use `ownyourjev.calibration.select_calibrator(p_dev, y_dev)`, which runs
   5-fold CV over none / temperature / platt-or-vector / isotonic and picks the lowest NLL, then refits on all of
   dev. You may restrict the candidate methods if you justify it (e.g. 200 examples are too few for isotonic on
   77 classes), but decide on dev.
   - Gold for Score heads: the level index, i.e. the gold score rounded to the nearest level.
   - Noul heads: binary probability of "true".
   - Calibrate every head of the task separately.
4. **Freeze before you score.** Write `benchmarks/results_v2/<task>/CALIBRATION.json` (format in
   `benchmarks/v2/calibration_eval.py`), including each head's `dev_cv` table and `n_fit`. Then run
   `uv run python -m benchmarks.v2.calibration_eval <task>` **exactly once**. It is the only code that reads
   test labels for this study, and it logs the call. Do not iterate on the test numbers.
5. **Stay in your lane.**
   - Do not touch `FROZEN.json`, `final.json`, the test probabilities, decisions or thresholds.
   - Do not edit `src/` or `benchmarks/*.py`.
   - Write only in `experiments/<task>/calibration/`, `data/v2/<task>/` and the two files above, plus your notes.
   - No git commits.
   - No LLM spend; this study needs none.
6. Log one ledger line with `benchmarks.v2.ledger.log_experiment(task=..., round=99, kind="selection",
   idea="post-hoc calibration", setting="calibration", dev=<dev_cv summary>, outcome=...)`.

## Deliverables in `benchmarks/results_v2/<task>/`
- `CALIBRATION.json`: written before scoring.
- `calibration.json`: written by `calibration_eval`.
- `dev_probs_deployed*.npy`: the out-of-sample dev predictions you used, one per head.
- `CALIBRATION_NOTES.md`:
  - how you obtained the dev predictions and how you verified them;
  - the dev CV table;
  - the chosen method and its parameters;
  - the test result vs Jev (ECE, adaptive ECE, Brier, NLL; raw, calibrated, and both oracle rows);
  - caveats.

## Tools
`ownyourjev.calibration` (`Calibrator`, `select_calibrator`, `cv_scores`, `calibration_metrics`), `ownyourjev.export.OnnxPredictor`,
`ownyourjev.models.decision.DecisionModel`, plus your task's own scripts in `experiments/<task>/`, which show
how the deployed model was built and run.

## Follow-up: serving consistency at batch size 1 (int8 tasks)
Dynamic int8 quantization computes activation scales over the whole input tensor. A text's probabilities
therefore depend on the other texts in its batch and their padding: agents measured differences of 0.13–0.29.
The deployed model serves one request at a time, so the reported test predictions must come from batch size 1.
1. Recompute the frozen model's test predictions at **batch size 1**: same artifact, preprocessing, threads,
   soup/fold logic, one text per call (`protocol.eval_inputs`, no labels). Save `eval_probs_b1[_<head>].npy`.
   Do the same for dev and save `dev_probs_b1[_<head>].npy`.
2. Refit the calibrator on the batch-1 dev predictions. Use the same rules: out-of-sample, 5-fold CV NLL, and
   argmax-preserving methods only (none/temperature) for Choice and Score heads.
3. Write `benchmarks/results_v2/<task>/BATCH1.json` (format in `benchmarks/v2/batch1_eval.py`), then run
   `uv run python -m benchmarks.v2.batch1_eval <task>` **exactly once**. It is logged.
4. Add a "Batch-1 serving" section to `CALIBRATION_NOTES.md`. Report the number of decision flips, the
   headline metric at batch 1 vs batched, the verdict at batch 1, and calibration at batch 1.
