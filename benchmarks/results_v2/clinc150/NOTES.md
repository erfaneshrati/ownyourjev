# clinc150 (v2): notes

**Test result (scored once, final.json):** accuracy **0.934** (95% CI 0.927 to 0.941) vs Jev-1.13 **0.895** on the same 5,500 test utterances (1,000 of them oos). Beats Jev; the CIs don't overlap.
**CPU latency:** p95 **23.8 ms**, p50 20.4 ms (ONNX Runtime, 4 threads, batch 1, n=300). Gate is < 70 ms.
**Label spend:** **$0.237** for this scope (deepseek-v4.1-flash only; cap was $1.40).
**Test reruns:** none.

## Final recipe
1. **Labels.** The whole pool (15,030 utterances) was labeled by deepseek/deepseek-v4.1-flash with a batched prompt
   (`experiments/clinc150/blabel.py`). One system prompt holds all 151 intent keys, each with a hand-written one-line
   description (`intents.py`), a few decision rules (read vs update intents, freeze_account vs account_blocked,
   distance vs directions, travel_notification vs travel_alert, what counts as oos) and the 20 seeds. Each request
   carries 20 utterances, which cuts the cost about 20x compared with one request per text (the prompt is about 3k tokens).
   Dev agreement is **0.965** (halves 0.96 / 0.97); v1 was ~0.90. The labeler marked 291 pool utterances oos (1.9%).
   No v1 silver labels were used.
2. **Synthetic oos.** 2,763 out-of-scope utterances were generated (`synth_oos.py`: 16 near-domain and far-domain topics × 6
   styles, temperature 1). The same labeler filtered them and kept the 2,366 it also called oos.
   2,066 went into training and 300 were held out.
3. **Student.** gte-modernbert full fine-tune, lr 5e-5, up to 4 epochs (early stopping on silver val picked epoch 3), max_len 64,
   bs 32, seed 0. Training data: 13,818 labeled pool texts (an 8% class-stratified silver-val split held out) plus 2,066 synthetic oos.
4. **oos threshold.** At inference, predict oos when the best in-scope probability is below 0.4.
5. **Export.** ONNX, with dynamic int8 applied only to the MLP input projections (`mlp.Wi`). Predictions match
   torch fp32 exactly on dev + silver val (1,412 texts). Model: `data/v2/clinc150/models/E_lr5_q_only_Wi/model.int8.onnx`.

## How choices were made (dev = 200 gold; "val" = 1,212 held-out silver; oos numbers pool dev's 8 + val's 23 real oos)
Dev alone is too small for oos (8 examples), so I estimated three things:
- in-scope accuracy on dev+val,
- oos recall on the 31 real oos (dev + val),
- oos recall on a harder held-out set of 250 "near-miss" synthetic oos (round 2, see below).

For the threshold, these were combined as 0.82·in-scope + 0.18·oos recall. That weighting uses the public CLINC test composition the brief gives (18% oos).

| run | in-scope acc (dev+val) | real oos recall (31) | near-miss oos recall | notes |
|---|---|---|---|---|
| A: pool only, lr 3e-5 | 0.959 | 0.81 (tau 0) / 0.90 (tau 0.4) | 0.67 / 0.80 | no synthetic oos |
| B: + 2,066 synth oos | 0.956 | 0.935 | 0.87 / 0.92 | synthetic oos: big oos gain |
| C: B + real oos ×3 | 0.954 | 0.90 | 0.84 | no gain |
| D: B + label smoothing 0.1 | 0.953 | 0.97 | 0.88 | in-scope down |
| **E: B with lr 5e-5, 4 ep** | **0.959** | 0.935 | 0.87 / **0.92 (tau 0.4)** | **chosen** |
| E seed 1 | 0.962 | 0.87 | 0.86 / 0.89 | seed noise ≈ ±2 oos examples |
| F: E + 879 near-miss synth oos (round 2) | 0.951 | 0.97 | (in-distribution) | in-scope −0.9 pt: some near misses really are in scope |

- Threshold: tau 0.4 maximised the combined objective for both E seeds (s0: 0.949 → 0.953, s1: 0.944 → 0.949).
  It costs about 0.1 pt of in-scope accuracy and adds 3 to 5 pt of near-miss oos recall.
  On test, the predicted oos share was 17.9% (an unlabeled statistic, seen after freezing), which is consistent with the 18% prior.
- Final dev, 200 gold: accuracy 0.965, in-scope 0.964, oos 8/8. The remaining dev errors are mostly ambiguous or noisy gold,
  for example "how many es are in receiving" is gold spelling, and "when do i have some time off next" is gold next_holiday.

## Latency / quantisation (dev texts, 4 threads; the machine was shared, load average 6 to 17)
- fp32 ONNX: p95 38 to 41 ms when the machine was quieter, but 65 to 99 ms under load. Too close to the gate.
- Full int8 (all MatMul/Gemm): p95 about 18 ms, but agreement with fp32 was only 0.977 on dev+val, and in-scope accuracy fell from 0.958 to 0.946 at tau 0.4.
- Partial int8 (`experiments/clinc150/partial_quant.py`), variant by variant:
  - Exclude the MLP output projections (mlp.Wo) and the head: agreement 0.9986, p95 19 ms.
  - Quantise only mlp.Wi: agreement **1.000**, p95 24 to 27 ms. **Chosen.**
  - Quantise only mlp.Wi + Wqkv: agreement 0.9986.
  - Keep the first and last 2 layers fp32: agreement 0.981.

  So **mlp.Wo is the int8-sensitive matrix in gte-modernbert**. This could become a `nodes_to_exclude` option in `ownyourjev.export.quantize_int8` (library feature to merge).

## What worked / didn't
- Worked:
  - Using the whole pool.
  - A batched labeling prompt with intent descriptions and rules: 0.965 dev agreement at $0.15 for the full pool.
  - LLM-generated oos with labeler filtering: real oos recall went from 0.81 to 0.94 and near-miss recall from 0.67 to 0.87, with in-scope accuracy roughly unchanged.
  - A mild confidence threshold.
  - Partial int8.
- Didn't:
  - Upweighting the real pool oos.
  - Label smoothing.
  - Near-miss synthetic oos as training data: it hurt in-scope accuracy.
  - Full int8: it cost about 1 pt.
- Not tried:
  - Synthetic in-scope data for weak intents: dev errors looked like label ambiguity, not weak classes.
  - Label cleaning with a second labeler.
  - Training on dev/seeds or on the silver-val split.
  - Seed ensembles: these exceed the latency budget unless int8.

## Next
- Retrain on 100% of the pool, with fixed epochs.
- Self-consistency or a second labeler on student/labeler disagreements.
- A 2-seed ensemble with partial int8 (about 50 ms p95) to reduce seed variance in oos recall.

## Files
- Code: `experiments/clinc150/`:
  - `blabel.py`, `intents.py`, `dev_label.py`, `label_pool.py`
  - `synth_oos.py`, `synth_oos2.py`
  - `train_student.py`, `compare.py`, `syn2_eval.py`
  - `partial_quant.py`, `predict_eval.py`, `export_check.py`
- Data: `data/v2/clinc150/`:
  - `pool_labeled.jsonl`, `synth_oos*.jsonl`
  - `models/*`
  - `eval_probs.npy`
