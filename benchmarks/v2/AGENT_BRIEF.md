# ownyourjev v2: experiment brief (read fully before starting)

## Goal
Prove you don't need Jev. For your task, build a model that **beats Jev's score on the same test
examples** while answering on **CPU in under 70 ms at p95** per request (4 threads, all of the task's
question heads together, because Jev answers them all in one call). The model runs on CPU; training
may use the GPUs.

## Hard rules (no cheating)
1. **Never look at test labels.** Use only `benchmarks/v2/protocol.py`: `seeds(task)` (20 gold),
   `dev(task)` (200 gold), `pool(task)` (unlabeled training texts), `eval_inputs(task)` (unlabeled test
   texts), `eval_folds(task)` (cross-fit tasks only). Do **not** load the Hugging Face datasets yourself,
   do not read `benchmarks/results/**` eval-score fields to steer choices, and do not import gold
   from `benchmarks/tasks.py`.
2. **Every decision is made on dev** (labeler model, prompt, label cleaning, data mix, backbone, epochs,
   thresholds, quantisation). The 220 gold examples are your whole human-label budget. You may also
   train on them, but then estimate on held-out dev folds, never on examples a model trained on.
3. **Score on test once.** When the recipe is frozen, write `benchmarks/results_v2/<task>/FROZEN.json`
   (format in `benchmarks/v2/final_eval.py`), then run
   `uv run python -m benchmarks.v2.final_eval <task>` **exactly once**. It is the only code that reads
   test labels and it logs every call to `benchmarks/results_v2/TEST_ACCESS.log`. If you must re-run,
   pass `--rerun-reason` and explain it in your NOTES.md. Do not iterate on the test score.
4. Only write inside `experiments/<task>/`, `data/v2/<task>/`, `benchmarks/results_v2/<task>/`.
   **Do not edit** `src/ownyourjev/**`, `benchmarks/*.py` or `benchmarks/v2/*.py`. Other agents run in
   parallel against them. If you need a library feature, implement it in your experiment folder
   (subclass or wrap), and describe it in NOTES.md so it can be merged later.
5. Do not commit or push to git.

## Reusing v1 LLM labels
`data/silver/<task>/silver.jsonl` and `silver_seeded.jsonl` hold v1 LLM labels (deepseek-v4.1-flash; plain
and with 20 v1 seed examples) for up to ~5,000 pool texts, keyed by `uid` (the same uids as `protocol.pool()`).
You may reuse their `labels` (and join them by uid), but **ignore their `gold` field**: gold beyond your
220 examples is off-limits. Drop any v1-labeled example whose uid is in your `seeds`/`dev`, and for SMS
Spam and SummEval also any uid outside `eval_inputs`. The v1 seeded labels used *different* seed examples
(drawn from the training pool); that is fine to reuse, but say so in NOTES.md.

## Budget (OpenRouter)
Run every script with `OYJ_BUDGET_SCOPE=<task> OYJ_SCOPE_CAP_USD=<your cap>`. The ledger enforces it,
along with a global $19 cap shared by all agents. `uv run ownyourjev budget` shows spend. LLM responses
are cached in `cache/llm_cache.db`, so re-runs are free. Try ideas on dev (200 examples cost cents) and
spend on full pools only when dev shows a win. Prices per 1M tokens (input/output):
deepseek/deepseek-v4.1-flash 0.057/0.144 (default; a reasoning model, the client disables reasoning),
qwen/qwen3.7-flash 0.03/0.13, google/gemini-2.5-flash-lite 0.10/0.40, anthropic/claude-haiku-5.5 0.10/0.50,
deepseek/deepseek-v4-pro 0.21/0.42, openai/gpt-5-mini 0.25/2.00, openai/gpt-5.6-luna 0.20/1.20,
google/gemini-3.1-flash-lite 0.25/1.50. Do **not** call any typesafe/jev model.

## Tools available (library, read the code)
- `ownyourjev.label.label_examples(client, questions, examples, purpose=..., seeds=..., guidance=...,
  distribution=True|False, system_preamble=...)`: labeling with extra guidelines (prompt optimisation)
  and soft labels (a probability per option, stored in `meta["soft"]`). The questions you pass for
  labeling may have richer descriptions than the task spec, as long as option keys and levels are unchanged.
- `ownyourjev.llm.OpenRouterClient(model=..., temperature=..., extra_body={"seed": i})`: other labelers,
  sampling, and self-consistency (the cache key includes extra_body).
- `ownyourjev.synth.synthesize(...)`: synthetic examples from the spec (optionally steered by seeds).
- `ownyourjev.train.train(question, train_ex, val_ex, TrainConfig(...))`: frozen / lora / full; with
  `soft_labels=True` it trains on soft targets. `device="auto"` waits for a free GPU slot (the GPUs are
  shared with other agents and the user's own jobs, so be patient and keep GPU memory modest).
- `ownyourjev.search.Search`: ladder and HPO (pass your dev-based `val` set; leave `test_fn=None`).
- `ownyourjev.train.predict_probs`, `ownyourjev.evaluate.evaluate` (on **dev** only).
- `ownyourjev.export.export_onnx` + `quantize_int8`; `uv run python -m benchmarks.v2.latency <task>
  <model_dir...> --out latency.json --onnx` (CPU, 4 threads, p95 is the gate).
- Backbones: `ultrabert`, `ultrabert-embed`, `gte-modernbert` (all ~150M). **No larger backbones.**

## What we already know (v1, LLM labels from deepseek-v4.1-flash, test numbers shown for context only)
- The labeler's agreement with gold bounds the student. Prompt examples (seeds) helped most:
  SMS Spam 0.920 → 0.956, Prompt Injections 0.767 → 0.905, Emotion 0.596 → 0.638.
- **CPU latency is binding.** PyTorch fp32 on 4 threads: 83–104 ms p95 for short texts, 219 ms for BoolQ.
  ONNX int8 brings gte-modernbert under the gate (AG News p95 46 ms, dev accuracy unchanged). **int8
  breaks UltraBERT** (BoolQ dev 0.82 → 0.64, even per-channel), and UltraBERT (24 layers) is slow on CPU
  even in fp32 (BoolQ p95 168 ms at ~270 tokens). Prefer gte-modernbert + int8 and check dev accuracy
  after quantising. Other latency levers: shorter `max_len` (check token lengths on dev), one shared
  encoder with several heads for multi-question tasks (write it in your folder), fewer threads per
  request is not allowed (gate is fixed at 4 threads).
- The machine is shared; latency measured while other agents run is pessimistic. Report what you get;
  the final latency will be re-measured on a quiet machine.

## Data-side levers to try (accuracy problems are data problems first)
- Prompt optimisation on dev: clearer label definitions, decision rules for confusable classes,
  dataset-style conventions learned from the 20 seeds and from dev error analysis (look at the labeler's
  dev mistakes, write guidelines that fix them, re-measure on dev; avoid overfitting dev: keep a held-out
  half of dev for the final comparison of prompt variants).
- Labeler choice: compare labelers' dev agreement; use a stronger model only where it pays.
- Label cleaning: two labelers or self-consistency; drop or down-weight disagreements; relabel the
  examples where the student and labeler disagree with a stronger model.
- Soft labels (`distribution=True`, `soft_labels=True`) for ordinal and noisy tasks.
- More, or better-targeted, training data: use the whole `pool` (the v1 run capped at 5,000), class
  balancing, synthetic hard examples for weak classes (steered by seeds and dev errors).
- Thresholds (noul F1 tasks): tune on dev; it is a frozen recipe parameter in FROZEN.json.

## Experiment ledger (for the paper): mandatory
Every idea you test, successful or not, gets one ledger line via `benchmarks.v2.ledger.log_experiment(...)`
(see the docstring for fields: kind, idea, change, dev metrics with the baseline you compared against,
cost, GPU minutes, outcome success/failure/neutral, one-sentence takeaway). Log as you go, not at the end.
Failures matter as much as successes. Never log test numbers there (final_eval records those).

## Re-scoring policy (rounds)
Round 1 = the first frozen recipe, scored once. A later round may be scored on test again only if its new
recipe beats the previous round's frozen recipe **on dev** by more than the dev noise (report the dev
comparison, e.g. a paired bootstrap on dev or mean over seeds), and must be run with
`final_eval --rerun-reason "round N: <what changed, dev evidence>"`. Every test look is reported in the paper.
final_eval archives the previous `final.json` / `FROZEN.json` automatically (do not copy them yourself).

## Deliverables in `benchmarks/results_v2/<task>/`
- `FROZEN.json` (written before scoring), `latency.json`, `final.json` (written by final_eval).
- `NOTES.md`: what you tried, with dev numbers for each idea, label spend, what worked and what didn't,
  the final recipe, CPU latency, the test result vs Jev, any test rerun and why, and anything you would
  try next. Keep it factual; negative results are valuable.
- Final model directories (with `model.int8.onnx` if used) under `data/v2/<task>/models/` (not in git).

## Steps 2 and 3: relaxed setting (approved by the user on 2026-10-19), reported separately in the paper
For tasks that did not beat Jev under the strict setting, two more levers are allowed, in this order.
Results are reported as a separate "relaxed setting" row next to the strict result, never instead of it.
- **Step 3, threads:** the latency gate is measured with **8 CPU threads**:
  `uv run python -m benchmarks.v2.latency <task> <model_dir> --out latency.json --onnx --threads 8`
  The gate is still CPU p95 < 70 ms for all of a task's questions per request.
- **Step 2, backbone:** encoders up to ~400M parameters are allowed if they pass the gate. `modernbert-large`
  (395M) is in the registry. Pre-step int8 ONNX p95 on 8 threads, on a loaded machine: ~44 ms at 32 tokens,
  ~64 ms at 64 tokens, ~100 ms at 128 tokens. So it is viable only for short inputs. At 8 threads, 150M models
  can afford longer inputs, more layers, or 2-model ensembles.
- **Data levers stay first:** reuse all labels and data you already have. LLM spend is limited to the scope cap
  in your message.
- **Rescoring:** the same dev-gated policy applies. The new frozen recipe must beat your current frozen recipe
  on dev beyond noise. Run `final_eval <task> --rerun-reason "relaxed (8 threads/<backbone>): <change + dev
  evidence>"`; the test look is counted and reported.
- **Ledger:** log experiments with `round` = the next round number and an extra field `setting="relaxed"`.
