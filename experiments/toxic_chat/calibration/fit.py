"""Fit per-head calibrators on dev (5-fold CV over none/temperature/platt/isotonic, lowest NLL), write
CALIBRATION.json. Also prints seed-robustness of the CV ranking (seeds 0-4) for the notes; selection uses seed 0."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import *
from ownyourjev.calibration import select_calibrator, cv_scores
dev = protocol.dev(T)
heads, summ = {}, {}
for h in HEADS:
    p = np.load(R / f"dev_probs_deployed_{h}.npy"); y = np.array([int(bool(e.gold[h])) for e in dev])
    cal, sc = select_calibrator(p, y)
    print(f"== {h}: n_pos {y.sum()} chosen {cal.method} {cal.params if cal.method != 'isotonic' else '(steps)'}")
    for m, s in sc.items():
        print(f"  {m:12s} nll {s['nll']:.4f} ece {s['ece']:.4f} ace {s['ace']:.4f} brier {s['brier']:.4f}")
    for seed in range(1, 5):
        s2 = cv_scores(p, y, seed=seed)
        print(f"  seed {seed}: " + " ".join(f"{m} {v['nll']:.4f}" for m, v in s2.items()), "-> best", min(s2, key=lambda m: s2[m]["nll"]))
    heads[h] = {**cal.to_dict(), "dev_cv": sc, "n_fit": int(len(y)), "n_pos": int(y.sum())}
    summ[h] = {m: round(s["nll"], 4) for m, s in sc.items()} | {"chosen": cal.method}
desc = ("Dev predictions: the frozen deployed artifact (soupAB_sq0.75/model.int8.onnx, eager attention + SmoothQuant 0.75 "
        "+ dynamic int8, batch 1, max_len 512, 8 threads; same path as predict_eval.py) on the 200 gold dev examples. "
        "The model never trained on dev or seeds (train_student.py blocks their uids; early stopping on silver pool val). "
        "Re-predicting 60 eval inputs matched the frozen test probs exactly (max |dp| 0). Per head, 5-fold stratified CV "
        "(seed 0) over none/temperature/platt/isotonic on dev, lowest out-of-fold NLL, refit on all 200.")
(R / "CALIBRATION.json").write_text(json.dumps({"description": desc, "heads": heads}, indent=1, default=float))
print(json.dumps(summ))
