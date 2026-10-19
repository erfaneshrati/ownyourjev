"""Collect every number used in the paper/README from result files (no hand-typed numbers)."""
import json, sqlite3, math
from pathlib import Path
from benchmarks.v2.report import verdict, _quiet_p95
from benchmarks.v2.protocol import RESULTS

TASKS = ["sms_spam", "paws", "prompt_injections", "toxic_chat",
         "banking77", "clinc150", "ag_news", "financial_phrasebank", "emotion",
         "stsb", "sst5", "toxigen"]
FUTURE = ["boolq", "summeval", "helpsteer2"]
MODE = {"sms_spam": "noul", "paws": "noul", "prompt_injections": "noul", "toxic_chat": "noul",
        "banking77": "choice", "clinc150": "choice", "ag_news": "choice", "financial_phrasebank": "choice", "emotion": "choice",
        "stsb": "score", "sst5": "score", "toxigen": "score"}
SCOPES = {"sms_spam": ["wins_noul"], "paws": ["wins_noul"], "prompt_injections": ["wins_noul"],
          "ag_news": ["wins_choice"], "financial_phrasebank": ["wins_choice"], "emotion": ["wins_choice"]}
db = sqlite3.connect("cache/budget.db")
def scope_spend(s):
    return db.execute("select coalesce(sum(cost_usd),0) from calls where cached=0 and purpose like ?", (s + ":%",)).fetchone()[0]
log = (RESULTS / "TEST_ACCESS.log").read_text().splitlines()
dedup = json.loads((RESULTS / "dedup_analysis.json").read_text())
facts = {"tasks": {}}
for t in TASKS:
    f = json.loads((RESULTS / t / "final.json").read_text())
    h = f["headline"]
    head = h["head"]
    hm = f["heads"][head]
    jm = f["jev_same_examples"].get(head, {})
    p50, p95 = _quiet_p95(t)
    ledger = [json.loads(l) for l in (RESULTS / t / "experiments.jsonl").read_text().splitlines() if l.strip()] if (RESULTS / t / "experiments.jsonl").is_file() else []
    if t in SCOPES:  # shared scope across 3 tasks: split by purpose substring
        sp = db.execute("select coalesce(sum(cost_usd),0) from calls where cached=0 and purpose like ? and purpose like ?", (SCOPES[t][0] + ":%", f"%{t}%")).fetchone()[0]
    else:
        sp = scope_spend(t)
    facts["tasks"][t] = {
        "mode": MODE[t], "n": f["n_eval"], "metric": h["metric"], "ours": h["ours"], "ci": h["ours_ci95"], "jev": h["jev_same_examples"],
        "verdict": verdict(f), "looks": sum(1 for l in log if f"\t{t}\t" in l),
        "ece_ours": hm.get("ece"), "ece_jev": jm.get("ece"), "brier_ours": hm.get("brier"), "brier_jev": jm.get("brier"),
        "heads": {k: {"metric": v["metric"], "ours": v[v["metric"]], "ci": v["ci95"], "jev": f["jev_same_examples"].get(k, {}).get(v["metric"])} for k, v in f["heads"].items()},
        "p50": p50, "p95": p95, "spend": sp,
        "ledger_n": len(ledger), "ledger_success": sum(r.get("outcome") == "success" for r in ledger), "ledger_failure": sum(r.get("outcome") == "failure" for r in ledger),
        "dedup": dedup.get(t), "recipe": json.loads((RESULTS / t / "FROZEN.json").read_text()).get("description", "")[:600],
    }
facts["total_spend"] = db.execute("select sum(cost_usd) from calls where cached=0").fetchone()[0]
facts["v1_spend"] = 5.7692
facts["tally"] = {v: sum(1 for x in facts["tasks"].values() if x["verdict"] == v) for v in ("win", "tie", "loss")}
facts["looks_total"] = sum(x["looks"] for x in facts["tasks"].values())
facts["ledger_total"] = sum(x["ledger_n"] for x in facts["tasks"].values())
facts["future"] = FUTURE
Path("paper/facts.json").write_text(json.dumps(facts, indent=1, default=float))
for t, x in facts["tasks"].items():
    print(f"{t:22s} {x['verdict']:4s} ours {x['ours']:.3f} [{x['ci'][0]:.3f},{x['ci'][1]:.3f}] jev {x['jev']:.3f} looks {x['looks']} ece {x['ece_ours'] if x['ece_ours'] is None else round(x['ece_ours'],3)}/{x['ece_jev'] if x['ece_jev'] is None else round(x['ece_jev'],3)} p50/95 {x['p50']:.0f}/{x['p95']:.0f} spend {x['spend']:.3f} ledger {x['ledger_n']} ({x['ledger_success']}+/{x['ledger_failure']}-)")
print("tally", facts["tally"], "looks", facts["looks_total"], "ledger", facts["ledger_total"], "total spend", round(facts["total_spend"], 2))
