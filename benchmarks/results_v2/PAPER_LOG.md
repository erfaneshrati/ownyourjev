# ownyourjev: research log for the paper

This is the coordinator's running record of the study: what was decided, when and why, every deviation and
incident, and where the evidence lives. Per-task experiment records are in `<task>/experiments.jsonl`
(the ledger) and `<task>/NOTES.md`. Every test-set scoring is in `TEST_ACCESS.log`.

## Research question
Can a ~150M-parameter encoder, trained on labels from a cheap LLM plus a small human-label budget, replace
TypeSafe's Jev (a hosted "System One" decision model) on its own benchmark suite? Replacing it means matching
or beating Jev's accuracy on the same test examples while running on CPU faster than Jev's best published
latency.

## Baseline
Jev-1.13.0 zero-shot results and raw per-example responses come from Deußer, Sparrenberg & Sifa (2026),
arXiv:2609.37647 (MIT code: AppliedMachineLearning-Lab/jev-benchmarking; responses on Zenodo
10.5281/zenodo.23039006). We use the same 15 datasets (5 per question type: noul, choice, score), splits and
question templates. Jev is re-scored from those responses on exactly our evaluation examples. We never call
the Jev API.

## Timeline and decisions
### v1 (2026-10-17/18): exploratory
- **Setup.** One labeler (deepseek/deepseek-v4.1-flash). Pool capped at 5,000. Model selection on held-out
  LLM labels. Three variants per task: LLM labels only; LLM labels with 20 gold examples in the prompt; gold
  labels as an upper bound.
- **Result.** 6 of 15 tasks beat Jev.
- **Known methodological problems,** which are why v2 exists:
  - The variant shown as the headline was chosen after seeing test scores.
  - Two code fixes were triggered by test numbers: the selection metric and the calibration guard (Prompt
    Injections), and the cross-fitting of secondary heads (SummEval).
  - Learning curves were scored on test.
  - No v1 model met the CPU latency gate below, which was introduced later.
- **Cost.** $5.77 of LLM labeling.
- v1 results stay in `benchmarks/results/` for reference. They are not used as paper results.

### v2 protocol (2026-10-18), decided before any v2 experiment
- **Human-label budget per task:** 20 seed examples plus 200 dev examples, all drawn from the training split.
  For SMS Spam and SummEval, which have no training split, they are carved out of the evaluation set and Jev
  is re-scored on the remainder. All other gold labels are hidden from experiments
  (`benchmarks/v2/protocol.py`).
- **Selection.** Every decision is made on dev: labeler, prompt, label cleaning, data mix, backbone, epochs,
  threshold, quantisation.
- **Test access.** Only `benchmarks/v2/final_eval.py` reads test labels, and every call is logged.
- **Gates,** decided with the user:
  - CPU only: per-request p95 < 70 ms on 4 threads, with all of a task's question heads answered together,
    as Jev answers them in one call. 70 ms is the bottom of Jev's published 70–500 ms end-to-end range.
  - GPU latency is not reported.
  - Backbones stay at ~150M parameters (UltraBERT-150M, UltraBERT-Embed-150M, gte-modernbert-base). Accuracy
    problems are tackled with data, not larger models.
- **Budget.** OpenRouter spend is capped at $20 in total, with per-task scope caps enforced by the ledger.
- **Rounds** (added after the user asked to keep going until every task beats Jev; policy fixed before any
  round-2 work):
  - Round 1 is the pre-registered result.
  - A later round may be scored on test again only if it beats the previous round on dev by more than the
    dev noise, and only with a logged `--rerun-reason`.
  - Earlier rounds' results are archived automatically.
  - The paper reports every test look.

### Execution
- **Agents.** 11 parallel agents, one per losing or tied task plus two that re-certify the existing wins.
  They all share `benchmarks/v2/AGENT_BRIEF.md`.
- **CPU latency findings,** made before the agents launched:
  - Plain PyTorch fp32 missed the gate even for short texts (83–104 ms p95).
  - ONNX Runtime with int8 made gte-modernbert pass with no dev loss (AG News p95 46 ms).
  - int8 broke UltraBERT (BoolQ dev 0.82 → 0.64).

## Incidents and deviations (complete list)
1. **GPU scheduler deadlock (2026-10-18 ~21:20).**
   - What happened: the shared scheduler made a process wait for free GPU memory while it held memory
     itself. All jobs stalled at 0% GPU for an unknown period, at most about 40 minutes.
   - Fix: a process's own reserved memory now counts as available, and after 120 s it takes any free slot.
   - Cost: 4 stuck jobs were killed (2 for Prompt Injections cross-validation, 2 for ToxicChat training).
     Their agents were asked to rerun them. One notice first reached the wrong agent (ToxiGen), which
     correctly ignored it; it was then resent.
   - No test data was involved.
2. **Slow budget check during labeling,** reported by the ToxiGen agent. A ~150 ms SQLite query ran on every
   request, which limited throughput. Agents worked around it locally; a library fix is pending.
3. **Latency measured on a loaded machine.** All latency numbers taken during agent runs came from a shared,
   heavily loaded machine (load average 9–28). Final latencies will be re-measured on a quiet machine before
   publication.

4. **Duplicate texts across splits** (found 2026-10-19, input texts only, no labels). Some datasets contain
   repeated inputs across their own splits:
   - SMS Spam: 82 of 5,354 evaluation texts (1.5%) exactly duplicate one of the 220 seed/dev texts, plus 31
     near-duplicates.
   - Prompt Injections: 4 of 116 evaluation texts are near-duplicates (>0.9 similarity) of training texts.
   - PAWS: no exact duplicates of training texts in the evaluation set.

   Planned robustness analysis: after all rounds finish, re-score every task with duplicated evaluation
   inputs excluded, logged as an analysis (not a model change) and reported next to the main numbers.
5. **OpenRouterClient reuse across asyncio event loops** silently turns calls into label failures (reported
   by the SMS/PAWS/PI agent). Workaround: one client per `asyncio.run`. Library fix pending.

6. **LLM cache key ignores `max_tokens`** (reported by the ToxiGen agent). A reply truncated by a too-small
   token limit stays cached when the limit is raised. Workaround: change the prompt. Library fix pending.
7. **Transductive shortcut declined** (ToxiGen round 2). Adding the unlabeled evaluation texts, with ensemble
   LLM labels, to the training set raised dev Spearman from 0.776 to 0.796. The protocol allows it, since
   those texts carry no gold labels, but it was not adopted because it says nothing about generalisation to
   new inputs. Only SMS Spam and SummEval train on evaluation texts, because they have no training split, and
   both are cross-fitted.
8. **Re-scoring gates held** (round 2): SST-5 (+0.005 dev, paired 95% CI −0.040..+0.050) and ToxiGen
   (AUROC +0.004, CI −0.001..+0.011) were not re-scored on test.

9. **Archive naming collision (STS-B).** The brief told agents to copy round files by hand, and final_eval
   also archived them automatically. As a result STS-B's `round2_final.json` / `round2_FROZEN.json` contain
   round 1 (identical to `round1_*`). The brief has been fixed so archiving is automatic only.
10. **Quiet-machine latency** (2026-10-19 ~01:40, load 0.6–2). Every final model was re-measured. 14 of 15 pass
    p95 < 70 ms. ToxicChat measured 68.5–74.8 ms because max_len is 512 and a few prompts are long (dev p95 178
    tokens); a round-2 latency fix was assigned. Numbers are in `_quiet_latency/` and `RESULTS.md`.

11. **Scope cap overshoot (ToxicChat, 2026-10-19 ~09:54).** The toxic_chat scope spent $1.771 against a
    $1.64 cap, $0.13 over. Cause: many concurrent batched calls to an expensive labeler (claude-sonnet,
    ~$0.01/call) each passed the budget check before any was recorded. This was made worse by the cached
    budget totals introduced earlier (5 s refresh). Fix: in-flight requests now reserve their estimated cost
    before sending. The agent was told to stop LLM calls. The total budget is unaffected ($24.80 cap).

## Results snapshot (2026-10-19 ~01:45, from RESULTS.md)
**8 wins, 4 ties, 3 losses** by the CI rule.
- **Wins:** SMS Spam, PAWS, Prompt Injections, Banking77, CLINC150, AG News, Financial PhraseBank, Emotion.
- **Ties:** ToxicChat, STS-B (after round 2), SST-5, ToxiGen. ToxiGen is effectively a loss: 0.854 vs 0.878,
  with the CI upper bound just touching Jev.
- **Losses:** BoolQ (after round 2), SummEval, HelpSteer2.
- **Test looks:** one per task, except BoolQ and STS-B, which have two.
- **Spend:** $17.41 of $20.

## Step 2/3 decision (2026-10-19, user)
For the stuck tasks the user approved, in this order: (1) up to $5 more LLM labeling (total budget $25);
(2) larger backbones if CPU p95 stays < 70 ms; (3) measuring the gate on 8 CPU threads instead of 4.

Backbone CPU latency grid, measured before step 2 started. Values are int8 ONNX p95 in ms, batch 1, on a
moderately loaded machine (load 4–6, so pessimistic). Raw data is in the scratchpad grid.

| backbone | threads | 32 tok | 64 tok | 128 tok | 256 tok |
|---|---|---|---|---|---|
| gte-modernbert (149M) | 4 | 22 | 30 | 56 | 108 |
| gte-modernbert (149M) | 8 | 19 | 32 | 44 | 76 |
| ModernBERT-large (395M) | 4 | 64 | 95 | 139 | 301 |
| ModernBERT-large (395M) | 8 | 44 | 64 | 100 | 183 |

Implications:
- A 395M encoder fits the gate only for inputs of about 64 tokens or fewer, and needs 8 threads to do so.
  That makes it viable for SST-5, STS-B and ToxiGen.
- It is not viable for BoolQ, SummEval or HelpSteer2 (long inputs), or for ToxicChat (p95 governed by
  ~180-token prompts).
- For those long-input tasks, 8 threads buys longer inputs or more depth at 150M instead.

## Round 3 / ToxicChat round 2 outcomes (2026-10-19)
- **STS-B round 3:** +11.7k re-paired pairs ($1.03). The dev gain over round 2 was +0.006, CI [−0.004, +0.016].
  Not re-scored. Gains per added pair are diminishing.
- **BoolQ round 3:** cross-fitted teacher distillation plus shared-init soups all reached ~0.88 dev, the same as
  round 2. Not re-scored. The 5-model prediction ensemble (0.90 dev) could not be distilled into one 150M
  model at 224 tokens. Capacity-limited.
- **ToxicChat round 2:** latency fixed (p95 ~44 ms) by exporting with eager attention, which removes the
  per-layer SDPA masking ops (15–18% of CPU time), plus SmoothQuant on `mlp/Wo`. Re-scored once, justified by
  the gate failure. Toxic F1 is still a tie (0.787 vs 0.786). Jailbreak F1 fell from 0.871 to 0.849 because
  its threshold was re-chosen; it still beats Jev's 0.717. Lowering max_len did not reduce p95, because the
  slowest requests are ~180-token prompts.

- **SST-5 step 1 (labeler probe, $1.27).** Nine labelers were tried, from $0.03/M models to frontier ones
  (gpt-5-mini, claude-sonnet-5.5, gemini-3.1-pro, gpt-5.6-terra, deepseek-v4-pro). On the held-out dev half they
  agree with gold at 0.52–0.60, against 0.570 for the current cheap labeler. A 5-labeler average reached 0.620,
  +0.05 with CI [−0.02, +0.12]. Reasoning did not help. Not re-scored. **Finding:** SST-5 is limited by the
  dataset's noisy, idiosyncratic labels, not by labeler strength.

- **ToxicChat round 3 (step 1, $0.74 incl. the overshoot).** Stronger labelers did not beat the cheap pair on
  dev F1. Sonnet ranked best (AP 0.883) but is more conservative: it flipped 100 of 400 positives. Students
  retrained on the relabeled data showed a paired bootstrap F1 difference of 0.000. Not re-scored. With only 16
  dev positives, dev cannot separate the candidates.
- **SummEval round 2 (step 1, $0.39).** Added 1,031 "natural" weak-LLM summaries (in-fold) as targets for the
  consistency head. Dev: +0.037, paired bootstrap over 14 articles CI [0.012, 0.066]. Re-scored (2nd test look):
  0.472 vs round 1 0.466 vs Jev 0.556. Consistency, the targeted dimension, fell on test (0.439 → 0.403) while
  coherence rose. A sentence-level fact-check signal could not be distilled into the CPU student (0.41 on its
  own target). Bookkeeping error: the round-2 FROZEN.json embeds round-1 dev and latency values; the correct
  values are in dev_metrics_round2.json and latency_round2.json.
- **Finding: dev-gated reruns still regress toward zero on test** (BoolQ +3.5 dev → +1.1 test; SummEval +3.7 →
  +0.6; STS-B +1.5 → +1.4 is the exception). Small dev sets (200 examples, or 14 articles) overestimate gains even
  when a paired CI excludes zero.

- **Relaxed setting: ToxicChat.** A 2-model prediction ensemble gave no dev gain (paired F1 difference 0.000) and
  was too slow (p95 241 ms under load). ModernBERT-large was about 4x slower on the ~180-token prompts that set
  p95, so it cannot meet the gate. Not re-scored.
- **Relaxed setting: ToxiGen.** ModernBERT-large (395M) trained on the 4-labeler ensemble soft labels matched the
  150M model on dev: AUROC +0.002 (CI −0.013 to +0.018), Spearman +0.006 (CI −0.029 to +0.042). int8 costs
  nothing at 395M, unlike at 150M. Not re-scored. **Revised finding:** ToxiGen is limited neither by labeler
  quality (round 2) nor by model size (round 3). The remaining gap is probably the gold definition
  (toxicity_ai + toxicity_human > 5.5) plus the dataset's annotation noise, which an LLM-labeled student does
  not reproduce.

- **HelpSteer2 round 2 (step 1, $0.86).** Labeler probe on dev (Spearman, helpfulness / correctness):
  flash without reasoning 0.388 / 0.410, flash with reasoning 0.458–0.484 / 0.426–0.439 (above Jev's test
  0.400 / 0.369), haiku 0.423 / 0.354, gemini-3.1-flash-lite 0.459 / 0.402, deepseek-v4-pro 0.364 / 0.355.
  Students on reasoning labels gained −0.002 [−0.036, +0.030], and +0.004 at 512 tokens. A pairwise ranking loss
  did not help. Flash labels on the remaining 12.8k pool scored −0.016 [−0.058, +0.024], with helpfulness and
  correctness getting worse. Not re-scored. **Finding:** this is the third time that better labels did not make
  a better student on helpfulness and correctness.
- **Relaxed setting: SST-5.** ModernBERT-large (max_len 80) averaged 0.588 dev over 4 runs, worse than
  UltraBERT-150M (0.603), and ran at p95 145 ms on 8 threads (under load). 2-model ensembles averaged 0.594. A
  shared-init soup reached 0.625 fp32, but only 0.600 after int8 (paired difference 0.000). Not re-scored. SST-5
  is label-limited at ~0.60–0.62 in every setting.
- **Relaxed setting: BoolQ.** On 8 threads a single model reaches only 1.3x the 4-thread speed (p95 48.6–53.5 ms).
  2-model ensembles: 253 ms as separate sessions, 92–94 ms as one ONNX graph (75 ms at max_len 160), so they fail
  the gate. Their dev gain is small (mean 0.886 for pairs, 0.888 for triples, vs 0.885), and longer max_len (up
  to 384) adds nothing. Not re-scored. BoolQ remains a loss under every allowed setting.
- **Relaxed setting: STS-B.** ModernBERT-large on the full round-3 data scored 0.9163 dev, no better than the
  gte-base runs (0.916–0.921), and ran at p95 94–129 ms on 8 threads, so it fails the gate. The only CI-positive
  candidate was a base+large prediction ensemble (+0.012, CI +0.0005..+0.025), chosen after several looks and
  too slow (p50 75 ms). Not re-scored; STS-B stays a tie.
- **Relaxed setting: SummEval.** All 22 layers or longer inputs (256/320 tokens) lowered dev by 0.02–0.08.
  2-3 model seed ensembles gained +0.008 / +0.012 with CIs touching 0, and as separate ONNX sessions on 8
  threads they hit p95 90–240 ms. The round-2 single model runs at p95 37–40 ms on 8 threads. Not re-scored.
  SummEval is data-limited (86 articles, ~0.60 labeler agreement), not compute-limited.
- **Finding: 8 CPU threads give little.** The speed-up is about 1.3x for one model, and concurrent sessions
  contend for the same threads. The relaxed gate therefore admits a ~400M model only for inputs up to ~64
  tokens, and admits no ensembles.

## Final status (2026-10-19 ~12:20)
**8 wins, 4 ties, 3 losses** against jev-1.13 on the same examples, by the 95% CI rule (`RESULTS.md`).
- **All 15 final models pass the strict gate:** CPU p95 < 70 ms on 4 threads, measured on a quiet machine.
  Marginal cases: ToxicChat 68 ms and SummEval 68 ms worst-case.
- **Test looks:** 19 in total. Each task was scored once, except BoolQ, STS-B, ToxicChat and SummEval
  (2 each). Every rescore was dev-gated, and none flipped a verdict to a win.
- **Relaxed setting (8 threads, backbones up to ~400M):** no dev-gated improvement for any task and no
  rescoring, so it adds no result rows.
- **Spend:** $21.74 of $25 ($20 original + $5 approved for step 1).

**Wins:** SMS Spam, PAWS, Prompt Injections, Banking77, CLINC150, AG News, Financial PhraseBank, Emotion.
**Ties:** STS-B, SST-5, ToxicChat, ToxiGen (point estimate below Jev).
**Losses:** BoolQ, SummEval, HelpSteer2.

**Duplicate-exclusion robustness** (`dedup_analysis.md`, logged as an analysis test look): frozen predictions
re-scored with evaluation inputs that exactly duplicate any seed, dev or pool text removed.
- All 8 wins hold with similar margins.
- **ToxicChat:** 307 of 5,083 duplicates. Ours 0.775 vs Jev 0.790 on the remainder, so the tie leans to Jev.
- **ToxiGen:** 177 of 940 duplicates. The gap widens to −0.034.
- Training on pool texts that also appear in test helps our models on those tasks; this should be stated in
  the paper.

**Where a small CPU model replaces Jev:** intent routing, topic, domain sentiment, emotion with a
dataset-specific convention, spam, prompt injection and paraphrase.

**Where it does not, and why:**
- **Label-limited:** SST-5; no labeler exceeds ~0.62 agreement with gold, from the cheapest to frontier models.
- **Data or definition-limited:** ToxiGen and SummEval; neither better labels nor a 2.6x larger model helped.
- **Capacity-limited under the 70 ms CPU gate:** BoolQ reading comprehension, HelpSteer2 helpfulness and
  correctness, and SummEval. Ensembles that would help do not fit the gate.


## Calibration study (2026-10-19)
- **Setup.** 12 agents, one per task, follow `benchmarks/v2/CALIBRATION_BRIEF.md`.
  - Post-hoc calibrators are fitted only on out-of-sample predictions of the deployed model on the 200 gold dev
    examples. Prompt Injections uses out-of-fold CV models, because its deployed model trained on dev.
  - The method is chosen by 5-fold CV NLL on dev and frozen in `CALIBRATION.json`.
  - Each task is scored once by the audited `benchmarks/v2/calibration_eval.py`, which logs to TEST_ACCESS
    with `analysis=calibration`.
  - Decisions and accuracy are unchanged.
  - For both ours and Jev, an *oracle* row recalibrates by 5-fold CV on the test labels. It is a like-for-like
    upper bound and never a reported model.
- **Mid-study rule (decided at ~13:44 on API-consistency grounds, not from test results).** Choice and Score
  heads may use only argmax-preserving calibrators: none or temperature. A Jev-compatible answer's
  `choice`/level must equal the argmax of its served probabilities, and vector or isotonic scaling can break
  that. Noul heads may use any monotone map (temperature, Platt, isotonic). The rule was raised by the Emotion
  agent on dev grounds (vector changed the top label on 4% of dev).
  - Two tasks had already been scored with vector scaling on a multiclass head: STS-B and ToxiGen
    (toxicity head).
  - STS-B was re-scored once under the rule with a logged `--rerun-reason`; its vector result is kept in
    `calibration_round1_vector.json` (test ECE 0.0545 raw → 0.0581, NLL 1.050 → 1.021, Jev 0.136 / 1.352).
    The STS-B agent raised the issue after seeing its first test result; disclosed in its notes.
  - ToxiGen toxicity was not re-scored: under the restricted set its dev CV picks `none` (NLL 1.0389 vs
    temperature 1.0399), so the compliant result is the already-scored raw row. AG News re-froze to
    temperature before its first scoring.
- **Oracle fix.** The first oracle rows could select vector/isotonic and so re-classified (e.g. Financial
  PhraseBank Jev oracle accuracy 0.837 vs raw 0.731). `calibration_eval.oracle` was restricted to the same
  allowed methods, and `benchmarks/v2/oracle_recompute.py` recomputed every task's oracle rows once (logged
  `analysis=oracle_recompute`; old values kept as `*_oracle_unrestricted`).
- **Incident: batch-dependent int8 outputs.** Agents re-predicting dev/test could not reproduce frozen test
  probabilities to 1e-4. Cause: dynamic int8 quantisation computes activation scales over the whole input
  tensor (padding included), so a text's probabilities depend on its batch-mates (max |Δp| up to 0.76).
  Eight int8 tasks had frozen test predictions computed in batches, while serving and latency are batch 1.
  - Fix: each agent recomputed test and dev predictions at batch size 1 with the same artifact, refitted the
    calibrator on batch-1 dev, and scored once with the audited `benchmarks/v2/batch1_eval.py`
    (`analysis=batch1_serving`).
  - No verdict changed. Headline batched → batch 1: SMS Spam 0.958 → 0.957, PAWS 0.884 → 0.881,
    Prompt Injections 0.957 → 0.966, Banking77 0.861 → 0.871, CLINC150 0.934 → 0.934, AG News 0.893 →
    0.893 (CI low 0.8857 vs Jev 0.885), Financial PhraseBank 0.794 → 0.818, SST-5 0.583 → 0.580 (tie).
  - Decision flips 0.3–1.5% on most tasks; Banking77 3.7%, SST-5 5.1%, Financial PhraseBank 9.5%. FPB's
    test gain is not matched on dev (0.850 → 0.830), so it is noise from fragile int8 decisions.
  - The main results table keeps the pre-registered batched numbers; the calibration table uses batch 1.
  - ToxicChat and STS-B were already batch 1; Emotion and ToxiGen are fp32.
- **Results** (`CALIBRATION.md`, paper Table 3 / Figure 3). With calibrators fitted on 200 dev examples,
  ours beats Jev on ECE 12/14 heads (median 0.027 vs 0.086), NLL 12/14, Brier 10/14.
  - Jev better: ToxiGen toxic (all metrics), CLINC150 (ECE/ACE/Brier; the out-of-scope rule caps the
    reported top probability at ~0.28 on 18% of inputs, invisible with 8 OOS dev examples), ToxicChat toxic
    (Brier/NLL only), ToxiGen toxicity (Brier only).
  - Our dev-calibrated model beats even the test-oracle recalibration of Jev on 7/14 heads in ECE and 9/14
    in Brier and NLL.
  - Lesson: evaluate a dynamically quantised model at the batch size it will be served at (or quantise
    statically).

## Cross-task findings so far (from round 1; details in each task's NOTES.md and ledger)
- **Batched labeling** (10–50 items per call) cuts labeling cost 10–20x with no loss in agreement (Banking77,
  CLINC150, STS-B).
- **Labeling guidance** written from seed examples and from labeler errors on half of dev was the largest
  single lever:
  - Banking77: 0.735 → 0.84 labeler accuracy.
  - SST-5: 0.53 → 0.62. The agent found the labeler almost never used "very positive" without guidance.
  - Emotion: 0.49 → 0.785. The labeler had to learn the dataset's convention: the label is the emotion
    word attached to "feel/feeling", regardless of negation.
  - On the held-out half of dev the gains are smaller but persist.
- **Stronger or more expensive labelers rarely beat a cheap labeler with good guidance** (Banking77, SST-5,
  BoolQ).
- **Labeling the full training pool beats the v1 cap of 5,000** (CLINC150 15k: 0.846 → 0.934; BoolQ).
- **Synthetic data** works when it targets a gap the real data doesn't cover; CLINC150 out-of-scope recall
  went from 0.81 to 0.935. It fails when it doesn't match the real distribution: BoolQ passage-driven
  questions were too easy and hurt the student.
- **Quantisation:**
  - Selective int8 (MLP output projection kept in fp32) keeps fp32 accuracy at near-int8 speed (CLINC150,
    STS-B, SST-5).
  - SmoothQuant plus a rewritten forward pass for ONNX export makes UltraBERT int8-viable and 1.8x faster
    (BoolQ).
- **Seed variance** on 200-example dev sets is often larger than the effects being measured. Weight-averaging
  several seeds ("model soup") removes most of it at no inference cost (BoolQ, ToxiGen).
- **Self-distillation on the unlabeled pool** (PAWS): two UltraBERT teachers trained on LLM labels pseudo-labeled
  ~41k unlabeled training pairs, and a student on soft pseudo-labels plus LLM labels reached 0.890 dev.
  gte-modernbert failed on PAWS (0.74 dev), where word order matters.
- **Better labels do not always transfer to the student.** ToxiGen: a 4-labeler ensemble lifted labeler
  Spearman from 0.754 to 0.818 on dev, but the student moved only from 0.770 to 0.776. SST-5 and SummEval
  look similar: the student caps near or below its labeler. Label quality is necessary but not sufficient
  on judgment-style tasks.
- **Synthetic negatives can teach the wrong cue.** SummEval: LLM-written summaries with deliberate errors
  hurt (0.487 → 0.445 dev), because real system differences are subtler than the planted ones.
- **Weight soups need a shared init.** STS-B: averaging runs that start from the same initial head and differ
  only in data order works (members 0.910–0.911, soup 0.911). Averaging independently seeded runs scores
  below its members.
- **Re-pairing pool sentences beats LLM-generated pairs** for STS-B (+0.012 seed-mean vs ±0). Generated
  pairs skew to high similarity.
- **A student cannot exceed a weak labeler; a strong labeler does not guarantee a strong student.** SST-5 is
  labeler-capped (~0.62). SummEval, HelpSteer2 helpfulness/correctness, ToxiGen and BoolQ are student-capped
  under the 150M / 70 ms CPU constraint. HelpSteer2 complexity (0.524 vs 0.460) beats Jev, where surface
  features suffice.
- **Dev sets drawn from training splits can be easier than test** (STS-B dev 0.910 vs test 0.879; Financial
  PhraseBank dev 0.85 vs test 0.794).

## Results table
Generated from `<task>/final.json` and `round*_final.json` by the report script (pending). Round-1 test
scorings so far are in `TEST_ACCESS.log`.
