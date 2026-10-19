from benchmarks.v2.ledger import log_experiment as L
T="sst5"
def l(**k): L(task=T, round=1, **k)
l(kind="labeling", idea="labeler comparison, 20 seeds, soft labels, no guidance", change="deepseek-flash/qwen3.7-flash/gemini-2.5-flash-lite/haiku-5.5 on dev",
  dev={"accuracy_deepseek_flash":0.530,"accuracy_qwen":0.550,"accuracy_gemini_lite":0.540,"accuracy_haiku":0.560,"n":200}, cost_usd=0.081, outcome="neutral",
  takeaway="all labelers below Jev; almost never predict 'very positive' (32/34 gold 4s -> 3)")
l(kind="labeling", idea="SST-convention guidance v1 written from dev-half-A labeler errors", change="guidance_v1.txt added to prompt",
  dev={"accuracy_deepseek_flash":0.620,"baseline_accuracy":0.530,"half_B":0.570,"half_B_baseline":0.530,"accuracy_qwen":0.595,"accuracy_haiku":0.555,"accuracy_gemini_lite":0.580,"n":200}, cost_usd=0.102, outcome="success",
  takeaway="biggest lever: guidance fixes under-use of 'very positive'; deepseek-flash best and cheap")
l(kind="labeling", idea="stronger labeler deepseek-v4-pro with guidance v1", change="labeler swap on dev",
  dev={"accuracy":0.615,"baseline_accuracy":0.620,"n":200}, cost_usd=0.154, outcome="failure", takeaway="7x the price, no gain over deepseek-flash")
l(kind="labeling", idea="average deepseek-flash and qwen soft labels", change="2-labeler mean on dev", dev={"accuracy":0.595,"baseline_accuracy":0.620,"n":200}, outcome="failure", takeaway="weaker second labeler dilutes")
l(kind="labeling", idea="blend v1 seeded silver labels (w=0.2) into deepseek guidance-v1 soft labels", change="dev items covered by v1 silver",
  dev={"accuracy":0.624,"baseline_accuracy":0.607,"n":117}, outcome="neutral", takeaway="+0.017 on 117 items is noise; not used")
l(kind="selection", idea="per-class log-bias calibration of labeler (5-fold CV on dev)", change="additive log-bias tuned on dev folds",
  dev={"accuracy_deepseek_v1":0.642,"baseline_accuracy":0.620,"accuracy_haiku_base":0.602,"haiku_baseline":0.560,"n":200}, outcome="neutral", takeaway="helps labeler argmax slightly; not applied to students (no gain there)")
l(kind="data", idea="label full pool with deepseek-flash + guidance v1 + seeds, soft", change="8,324 pool texts labeled", cost_usd=0.430, outcome="success", takeaway="full pool labeled cheaply thanks to provider prompt caching")
l(kind="training", idea="gte-modernbert on v1 seeded hard silver (4.9k)", change="baseline student", dev={"accuracy":0.535,"n":200}, gpu_minutes=5, outcome="failure", takeaway="v1-style labels give a sub-Jev student")
l(kind="training", idea="gte-modernbert soft vs hard targets on guidance-v1 labels", change="soft_labels=True vs False",
  dev={"accuracy_soft":0.585,"accuracy_hard":0.550,"n":200}, gpu_minutes=10, outcome="success", takeaway="soft targets help (+0.035 single run)")
l(kind="training", idea="gte-modernbert HPO/seeds (lr 2e-5..5e-5, 3-5 epochs, 3 seeds, mean pooling)", change="9 runs",
  dev={"accuracy_mean":0.56,"accuracy_min":0.540,"accuracy_max":0.585,"mean_pooling":0.570,"n":200}, gpu_minutes=45, outcome="neutral", takeaway="seed noise +-0.02; gte plateaus ~0.56")
l(kind="training", idea="add 220 gold (x3) to gte training, 2-fold cross-fit on dev", change="gold in train", dev={"accuracy":0.570,"baseline_accuracy":0.56,"n":200}, gpu_minutes=10, outcome="neutral", takeaway="no clear gain")
l(kind="data", idea="3.5k SST-style sub-phrases from pool sentences, teacher-labeled", change="+3,500 phrase examples (gte, 3 seeds)",
  dev={"accuracy_mean":0.550,"baseline_accuracy":0.56,"n":200}, cost_usd=0.172, gpu_minutes=20, outcome="failure", takeaway="phrase-level transfer data did not help the sentence-level student")
l(kind="selection", idea="student bias calibration (dev CV, or matching teacher on silver val)", change="additive log-bias", dev={"delta_range":[-0.03,0.03],"n":200}, outcome="failure", takeaway="no consistent gain")
l(kind="training", idea="UltraBERT backbone, soft labels", change="ultrabert lr 2e-5 seeds 0,1,2 + lr 1e-5",
  dev={"accuracy_runs":[0.600,0.615,0.610,0.610],"accuracy_mean":0.609,"baseline_accuracy":0.56,"n":200}, gpu_minutes=60, outcome="success", takeaway="UltraBERT +0.05 over gte on this task")
l(kind="training", idea="2-model UltraBERT prob ensembles", change="pairwise averages", dev={"accuracy_mean":0.614,"baseline_accuracy":0.609,"n":200}, outcome="neutral", takeaway="noise-level gain; would double latency")
l(kind="quantisation", idea="full dynamic int8 (gte / UltraBERT)", change="quantize_int8 per-channel",
  dev={"gte_accuracy":0.495,"gte_fp32":0.545,"ultrabert_accuracy":0.310,"ultrabert_fp32":0.600,"n":200}, outcome="failure", takeaway="int8 breaks UltraBERT and costs gte 0.05")
l(kind="quantisation", idea="partial int8: keep MLP down_proj, attention-score MatMuls and head in fp32", change="nodes_to_exclude",
  dev={"accuracy_ub_s0":0.590,"accuracy_ub_s1":0.600,"fp32_ub_s1":0.615,"n":200,"p95_ms":29.4}, outcome="success", takeaway="UltraBERT int8 works when down_proj stays fp32; p95 ~28 ms")
l(kind="latency", idea="UltraBERT fp32 ONNX / torch latency", change="no quantisation", dev={"p95_ms_onnx_fp32":89.0,"p95_ms_torch":439.0}, outcome="failure", takeaway="fp32 UltraBERT fails the 70 ms gate")
l(kind="quantisation", idea="partial int8: quantise only attention projections", change="nodes_to_exclude everything but q/k/v/o", dev={"accuracy":0.610,"p95_ms":71.2,"n":200}, outcome="failure", takeaway="accurate but misses the gate")
l(kind="selection", idea="round-1 frozen recipe: UltraBERT seed1 + partial int8", change="frozen", dev={"accuracy":0.600,"spearman":0.857,"n":200}, cost_usd=0.94, outcome="neutral", takeaway="test parity with Jev (see final.json)")
