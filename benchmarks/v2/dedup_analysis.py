"""Robustness analysis (not a model change): re-score each task's frozen predictions with evaluation inputs
that exactly duplicate (after normalisation) any text the experiment could see -- seeds, dev and the training
pool (for cross-fit tasks: seeds and dev only, since the pool *is* the unlabeled eval set) -- removed.
Jev is re-scored on the same subset. Reads test labels, so it is logged to TEST_ACCESS.log as 'analysis'."""

from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np

from benchmarks.run import jev_answers, jev_probs_matrix
from benchmarks.tasks import MODES, get_task
from benchmarks.v2 import protocol as P
from benchmarks.v2.final_eval import HEADLINE, _metric_for
from ownyourjev.evaluate import binary_metrics, evaluate
from ownyourjev.spec import render_state


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def main() -> None:
    out, lines = {}, []
    for t in [x for m in ("noul", "choice", "score") for x in MODES[m]]:
        fz = P.RESULTS / t / "FROZEN.json"
        if not (P.RESULTS / t / "final.json").is_file():
            continue
        frozen = json.loads(fz.read_text())
        spec = get_task(t).spec
        s = P._splits(t)
        seen = s["seeds"] + s["dev"] + ([] if t in P.CROSS_FIT else s["pool"])
        seen_set = {norm(render_state(e.state)) for e in seen}
        ev = s["eval"]
        keep = np.array([norm(render_state(e.state)) not in seen_set for e in ev])
        groups = [e.group for e in ev] if spec.grouped else None
        res = {"n_eval": len(ev), "n_dup": int((~keep).sum())}
        ours_h, jev_h = {}, {}
        for h, cfg in frozen["heads"].items():
            q = spec.parsed[h]
            p = np.load(cfg["probs"])[keep]
            golds = [e.gold[h] for e, k in zip(ev, keep) if k]
            gr = [g for g, k in zip(groups, keep) if k] if groups else None
            name = _metric_for(t, h, q)
            m = binary_metrics(golds, p, threshold=float(cfg.get("threshold", 0.5))) if q.type == "noul" else evaluate(q, golds, p, gr)
            ours_h[h] = m[name]
            jm = jev_probs_matrix(q, jev_answers(spec.questions, ev, h), ev)
            if jm is not None:
                jev_h[h] = evaluate(q, golds, jm[keep], gr)[name]
        head, metric = HEADLINE.get(t, (spec.primary_head, None))
        if head == "*mean*":
            res["ours"], res["jev"] = float(np.mean(list(ours_h.values()))), float(np.mean(list(jev_h.values())))
        else:
            res["ours"], res["jev"] = ours_h[head], jev_h.get(head)
        out[t] = res
        lines.append(f"| {t} | {res['n_dup']} / {res['n_eval']} | {res['ours']:.4f} | {res['jev']:.4f} | {res['ours'] - res['jev']:+.4f} |")
    (P.RESULTS / "dedup_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    table = "\n".join(["| task | duplicated eval inputs removed | ours | Jev | ours − Jev |", "|---|---|---|---|---|", *lines])
    (P.RESULTS / "dedup_analysis.md").write_text("# Duplicate-exclusion robustness analysis\n\n" + (__doc__ or "") + "\n\n" + table + "\n")
    with (P.RESULTS / "TEST_ACCESS.log").open("a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\tALL\tanalysis=dedup_exclusion\t(no model change; frozen predictions re-scored on de-duplicated subsets)\n")
    print(table)


if __name__ == "__main__":
    main()
