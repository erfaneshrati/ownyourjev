"""One-off backfill of the experiment ledger (round 1) from NOTES.md / logs. Dev numbers only; no test numbers."""
from benchmarks.v2.ledger import log_experiment as L

T = "banking77"
def lg(**k): L(task=T, round=1, backfilled=True, **k)

base_note = "v1 silver (one query per call, no descriptions) agreed with gold on 0.735 of 102 overlapping dev examples"
# --- labeler prompt / model comparisons (prompt: rich descriptions v1 + rules + 20 seeds, batch 20) ---
lg(kind="labeling", idea="rich intent descriptions + decision rules + 20 seeds, batched 20 queries/call (deepseek-v4.1-flash)",
   change="labeling prompt only; option keys unchanged", dev={"accuracy": 0.81, "accA": 0.77, "accB": 0.85, "baseline_accuracy": 0.735, "baseline": base_note, "n": 200},
   cost_usd=0.0063, outcome="success", takeaway="descriptions + rules lift labeler agreement by ~7 points over v1 silver; batching makes it cheap")
for m, acc, a, b, c, out, tk in [
    ("deepseek/deepseek-v4-pro", 0.81, 0.79, 0.83, 0.0587, "neutral", "no better than flash at ~10x the cost"),
    ("anthropic/claude-haiku-5.5", 0.805, 0.79, 0.82, 0.0062, "neutral", "on par with flash"),
    ("openai/gpt-5-mini", 0.77, 0.75, 0.79, 0.0059, "failure", "worse than flash (minimal reasoning)"),
    ("google/gemini-3.1-flash-lite", 0.81, 0.77, 0.85, 0.0126, "neutral", "on par with flash, 2x cost"),
    ("openai/gpt-5.6-luna", 0.765, 0.72, 0.81, 0.0095, "failure", "worst; 20/200 answers missing in batched JSON"),
    ("qwen/qwen3.7-flash", 0.79, 0.76, 0.82, 0.0014, "neutral", "slightly worse than flash but cheapest; useful as second voter"),
    ("google/gemini-2.5-flash-lite", 0.79, 0.77, 0.81, 0.0045, "neutral", "slightly worse than flash")]:
    lg(kind="labeling", idea=f"labeler choice: {m}", change="same prompt as flash (descriptions v1 + 20 seeds, batch 20)",
       dev={"accuracy": acc, "accA": a, "accB": b, "baseline_accuracy": 0.81, "baseline": "deepseek-v4.1-flash same prompt", "n": 200},
       cost_usd=c, outcome=out, takeaway=tk)
lg(kind="labeling", idea="majority vote of labelers (seeds-only prompt)", change="5-labeler vote (flash, pro, haiku, gemini-3.1, qwen) and 3-labeler vote (flash, haiku, gemini-3.1)",
   dev={"accuracy_vote5": 0.825, "accuracy_vote3": 0.825, "unanimous5_n": 160, "unanimous5_acc": 0.894, "baseline_accuracy": 0.81, "n": 200},
   cost_usd=0.0, outcome="neutral", takeaway="+1.5 points, within dev noise; errors are mostly shared dataset-convention ambiguities")
# --- more gold in prompt (seeds + dev half A, scored on half B) ---
for m, b0, b1, c in [("deepseek/deepseek-v4.1-flash", 0.85, 0.86, 0.0044), ("google/gemini-3.1-flash-lite", 0.85, 0.86, 0.0175),
                     ("qwen/qwen3.7-flash", 0.82, 0.83, 0.0021), ("anthropic/claude-haiku-5.5", 0.82, 0.86, 0.0094)]:
    lg(kind="labeling", idea=f"add dev half A (100 gold) to labeling prompt examples: {m}", change="prompt examples: 20 seeds -> 20 seeds + 100 dev-A; scored on unseen dev half B",
       dev={"accB": b1, "baseline_accB": b0, "n": 100}, cost_usd=c, outcome="success" if b1 - b0 >= 0.03 else "neutral",
       takeaway="more gold examples in the prompt teach dataset conventions; small but consistent gain")
# --- rules v2 + batch 50 (seeds + dev A, scored on B) ---
for m, b, b0, c in [("deepseek/deepseek-v4.1-flash", 0.85, 0.86, 0.0045), ("google/gemini-3.1-flash-lite", 0.86, 0.86, 0.0105),
                    ("qwen/qwen3.7-flash", 0.79, 0.83, 0.0012), ("anthropic/claude-haiku-5.5", 0.82, 0.86, 0.0049)]:
    lg(kind="labeling", idea=f"guidelines v2 (5 extra rules from dev-A errors) + batch 50: {m}", change="added rules: fiat currency, spare vs physical card, country_support, declined vs acceptance, expiring card; batch 20->50",
       dev={"accB": b, "baseline_accB": b0, "n": 100}, cost_usd=c, outcome="neutral" if b >= b0 else "success",
       takeaway="no measurable gain on held-out half B (n=100 too noisy to separate); kept v2 rules")
# --- 2-fold dev CV (seeds + other half in prompt) ---
for m, bsz, acc, c in [("deepseek/deepseek-v4.1-flash", 20, 0.84, 0.0037), ("deepseek/deepseek-v4.1-flash", 50, 0.84, 0.0073),
                       ("qwen/qwen3.7-flash", 20, 0.81, 0.0022), ("qwen/qwen3.7-flash", 50, 0.82, 0.0012),
                       ("anthropic/claude-haiku-5.5", 20, 0.81, 0.0097), ("anthropic/claude-haiku-5.5", 50, 0.805, 0.0050),
                       ("google/gemini-3.1-flash-lite", 50, 0.835, 0.0106), ("deepseek/deepseek-v4-pro", 50, 0.835, 0.0421),
                       ("google/gemini-2.5-flash-lite", 50, 0.81, 0.0034)]:
    lg(kind="labeling", idea=f"2-fold dev CV, v2 prompt with seeds + other dev half: {m}, batch {bsz}",
       change="estimate of 'gold in prompt' labeling on all 200 dev", dev={"accuracy": acc, "baseline_accuracy": 0.81, "baseline": "flash, seeds-only prompt", "n": 200},
       cost_usd=c, outcome="success" if acc >= 0.835 else "neutral",
       takeaway="flash best and cheapest (0.84); batch 50 = batch 20" if "v4.1" in m else "not better than flash")
lg(kind="labeling", idea="majority vote of 3 labelers on dev CV labels", change="best 3-way vote over flash/qwen/haiku/gemini-3.1/pro/gemini-2.5 combos",
   dev={"accuracy_best_vote3": 0.845, "baseline_accuracy": 0.84, "n": 200}, cost_usd=0.0, outcome="neutral", takeaway="voting adds ~0.5 point at best: labelers share errors")
lg(kind="labeling", idea="cascade: flash+qwen on all, gemini-3.1-flash-lite + deepseek-v4-pro adjudicate disagreements",
   change="majority over 4 labelers on disagreements, tie-break pro>flash>gemini>qwen", dev={"accuracy": 0.835, "baseline_accuracy": 0.84, "agree_n": 185, "agree_acc": 0.87, "n": 200},
   cost_usd=0.0, outcome="neutral", takeaway="adjudication by stronger labeler does not beat flash alone on dev; kept as soft vote distribution")
# --- pool labeling ---
lg(kind="data", idea="label full pool (9,773) with flash and qwen, prompt = descriptions v2 + all 220 gold, batch 50",
   change="full pool instead of v1's <=5,000", dev={"flash_qwen_disagree": 840, "n_pool": 9773}, cost_usd=0.2661, outcome="success",
   takeaway="full pool labeled by two models for $0.27 thanks to batching + prompt caching")
lg(kind="data", idea="relabel the 840 flash/qwen disagreements with gemini-3.1-flash-lite and deepseek-v4-pro",
   change="adjudication labels for vote / soft labels", dev={"n_relabeled": 840, "hard_labels_changed_vs_flash": 298}, cost_usd=0.2239, outcome="neutral",
   takeaway="pro was the expensive part ($0.18); student gain from it is within dev noise")
# --- students ---
lg(kind="training", idea="pipeline check: gte-modernbert full FT on v1 seeded silver (4.9k)", change="baseline student",
   dev={"accuracy": 0.775, "accuracy_int8": 0.745, "n": 200}, gpu_minutes=0.8, outcome="neutral", takeaway="v1-label student ~0.78; int8 per-channel costs 3 points here")
lg(kind="quantisation", idea="int8 per-tensor vs per-channel dynamic quantisation (v1silver model)", change="per_channel=False",
   dev={"accuracy_int8_per_tensor": 0.73, "accuracy_int8_per_channel": 0.745, "accuracy_fp32": 0.775, "p95_ms_per_tensor": 21.9, "p95_ms_per_channel": 24.1, "p95_ms_fp32_onnx": 40.8, "n": 200},
   outcome="failure", takeaway="per-tensor slightly worse and barely faster; keep per-channel")
lg(kind="training", idea="gte-modernbert full FT on flash pool labels (9.8k)", change="labels: v1 silver -> flash v2 full pool",
   dev={"accuracy": 0.89, "baseline_accuracy": 0.775, "n": 200, "note": "optimistic: dev was in labeling prompt"}, gpu_minutes=15.3, outcome="success",
   takeaway="better labels + full pool: +11.5 points over v1-silver student (dev partly optimistic)")
lg(kind="training", idea="hard majority-vote labels (flash/qwen + gemini/pro adjudication)", change="labels: flash -> vote_hard",
   dev={"accuracy": 0.90, "devA": 0.88, "devB": 0.92, "baseline_accuracy": 0.89, "n": 200}, gpu_minutes=3.6, outcome="neutral", takeaway="+1 point, within noise")
lg(kind="training", idea="soft vote labels (vote distribution as target, soft_labels=True)", change="labels: vote_hard -> vote_soft",
   dev={"accuracy": 0.905, "devA": 0.87, "devB": 0.94, "accuracy_int8": 0.895, "baseline_accuracy": 0.90, "n": 200}, gpu_minutes=6.7, outcome="neutral",
   takeaway="best dev number but within noise of hard/flash-only; chosen as final for encoding labeler uncertainty")
lg(kind="labeling", idea="clean estimate: relabel pool with flash using only seeds + dev half A in prompt", change="pool labels without dev half B seen anywhere",
   dev={}, cost_usd=0.1806, outcome="neutral", takeaway="needed for an uncontaminated dev estimate")
lg(kind="training", idea="clean estimate student: flash devA-prompt labels, selected on dev A, scored on unseen dev B",
   change="val = dev A only", dev={"accB": 0.83, "accA": 0.88, "labeler_cv_accuracy": 0.84, "n_B": 100}, gpu_minutes=17.7, outcome="neutral",
   takeaway="uncontaminated estimate ~0.83: student roughly matches labeler accuracy, does not denoise beyond it")
lg(kind="training", idea="add dev half A gold x3 to student training", change="train += dev A x3, val = dev A",
   dev={"accB": 0.83, "baseline_accB": 0.83, "n_B": 100}, gpu_minutes=8.6, outcome="neutral", takeaway="100 gold examples among 9.8k silver do not move held-out accuracy")
lg(kind="latency", idea="final votesoft model, int8 ONNX, 4 threads, batch 1 (benchmarks.v2.latency)", change="",
   dev={"cpu_p50_ms": 14.6, "cpu_p95_ms": 24.5, "gate_ms": 70, "accuracy_int8": 0.895, "accuracy_fp32": 0.905}, outcome="success",
   takeaway="gte-modernbert int8 at max_len 64 is ~3x under the gate, shared machine")
lg(kind="selection", idea="freeze recipe: vote_soft labels + gte-modernbert full FT + int8 ONNX", change="FROZEN.json written before test scoring",
   dev={"accuracy_int8": 0.895, "clean_devB": 0.83, "n": 200}, cost_usd=0.0, outcome="success",
   takeaway="chosen on dev; total label spend $0.916 for the task (incl. all dev experiments)")
