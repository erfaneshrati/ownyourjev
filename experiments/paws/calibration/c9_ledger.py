"""Ledger line for the dev-only calibration selection (values from the frozen CALIBRATION.json; no test numbers)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from benchmarks.v2.ledger import log_experiment
h = json.loads((ROOT / "benchmarks/results_v2/paws/CALIBRATION.json").read_text())["heads"]["answer"]
summ = {m: {k: round(s[k], 4) for k in ("nll", "ece", "brier")} for m, s in h["dev_cv"].items()}
log_experiment(task="paws", round=99, kind="selection", idea="post-hoc calibration", setting="calibration",
               dev={"chosen": h["method"], "params": h["params"], "dev_cv": summ, "n_fit": h["n_fit"]}, outcome="success",
               takeaway=f"platt chosen by 5-fold CV NLL on dev ({summ['platt']['nll']} vs none {summ['none']['nll']}); test probs not reproducible to 1e-4 (int8 batch noise), selection robust to it")
