"""Shared helpers for the toxigen v2 experiment (labeling prompts, dev scoring, IO)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats

from benchmarks.v2 import protocol as P
from ownyourjev.data import Example
from ownyourjev.spec import Question

TASK = "toxigen"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "v2" / TASK
LAB = DATA / "labels"
RES = ROOT / "benchmarks" / "results_v2" / TASK
SPEC = P.task_spec(TASK)
QS: dict[str, Question] = SPEC.parsed

GUIDANCE_V1 = """
These texts come from ToxiGen: machine-generated statements that mention a minority or identity group
(Black, Asian, Latino/Mexican, Native American, Jewish, Muslim, Chinese, Middle Eastern, women, LGBTQ+,
people with physical or mental disabilities, ...). Human annotators rated how harmful each statement is to
the group it mentions, averaged over three people. Rate as such an annotator would.
- Merely mentioning a group, describing history, facts, discrimination against the group, or arguing
  against prejudice is benign (level 0), even if the topic is sensitive.
- Implicit and coded hate counts as much as explicit slurs: stereotypes stated as facts, "jokes", questions
  that insinuate ("why do they always ..."), dehumanising comparisons, saying a group does not belong, should
  go back, is a threat, burden, criminal, dirty, inferior, or should lose rights. Do not require profanity.
- Condescending, othering or patronising remarks, and "positive" stereotypes stated as generalisations,
  are usually mildly offensive (level 1-2), not benign.
- Insults or profanity not aimed at a group, and off-topic text, are at most slightly offensive.
- `toxic` is true when the average annotator would find the text clearly offensive toward a group,
  i.e. moderately offensive or worse (level 2+). Mild insensitivity alone is not toxic.
- Be consistent: `toxic` should agree with the `toxicity` level you give.
"""


def soft_seed_labels(ex: Example) -> dict:
    """Gold (mean rating 0-4, binary) -> distribution-style seed answer consistent with the gold."""
    g = float(ex.gold["toxicity"])
    lo = int(np.floor(g + 1e-6)); hi = min(4, lo + 1); w = round(g - lo, 2)
    dist = {str(i): 0.0 for i in range(5)}
    dist[str(lo)] += round(1 - w, 2)
    if w > 0:
        dist[str(hi)] += w
    return {"toxic": 1.0 if ex.gold["toxic"] else 0.0, "toxicity": dist}


def hard_seed_labels(ex: Example) -> dict:
    return {"toxic": bool(ex.gold["toxic"]), "toxicity": int(round(float(ex.gold["toxicity"])))}


def seeds(soft: bool = True) -> list[Example]:
    return [replace(e, labels=(soft_seed_labels(e) if soft else hard_seed_labels(e))) for e in P.seeds(TASK)]


def save(exs: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in exs:
            f.write(json.dumps({"uid": e.uid, "state": e.state, "labels": e.labels, "meta": e.meta}) + "\n")


def load(path: Path) -> dict[str, dict]:
    return {r["uid"]: r for r in (json.loads(l) for l in path.open())}


def soft_of(rec: dict) -> tuple[float, np.ndarray] | None:
    """-> (P(toxic), distribution over 5 levels) from a labeled record (soft if present, else one-hot)."""
    if rec.get("labels") is None:
        return None
    s = (rec.get("meta") or {}).get("soft")
    if s:
        return float(s["toxic"]), np.asarray(s["toxicity"], dtype=float)
    d = np.zeros(5); d[int(rec["labels"]["toxicity"])] = 1
    return float(bool(rec["labels"]["toxic"])), d


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cands = np.unique(scores)
    best = (0.5, -1.0)
    for i, t in enumerate(cands):
        acc = float(((scores >= t) == y).mean())
        if acc > best[1] + 1e-12:
            best = (float(t), acc)
    # midpoint between this candidate and the previous one, for robustness
    return best


def dev_report(p_tox: np.ndarray, dist: np.ndarray, dev: list[Example], idx=None) -> dict:
    y = np.array([bool(e.gold["toxic"]) for e in dev]); g = np.array([float(e.gold["toxicity"]) for e in dev])
    if idx is not None:
        y, g, p_tox, dist = y[idx], g[idx], p_tox[idx], dist[idx]
    es = dist @ np.arange(5)
    t_es, a_es = best_threshold(es, y)
    t_p, a_p = best_threshold(p_tox, y)
    comb = 0.5 * p_tox + 0.5 * es / 4
    t_c, a_c = best_threshold(comb, y)
    return {
        "n": int(len(y)),
        "acc_p@0.5": float(((p_tox >= 0.5) == y).mean()),
        "acc_es@2": float(((es >= 1.5) == y).mean()),
        "acc_p_best": a_p, "thr_p": t_p,
        "acc_es_best": a_es, "thr_es": t_es,
        "acc_comb_best": a_c,
        "spearman_es": float(stats.spearmanr(g, es).statistic),
        "spearman_p": float(stats.spearmanr(g, p_tox).statistic),
        "auroc_p": float(__import__("sklearn.metrics").metrics.roc_auc_score(y, p_tox)) if len(set(p_tox)) > 1 else float("nan"),
        "auroc_es": float(__import__("sklearn.metrics").metrics.roc_auc_score(y, es)),
    }
