"""Shared helpers for the summeval v2 experiment."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

from benchmarks.v2 import protocol as P
from ownyourjev.spec import Question

T = "summeval"
DIMS = ["coherence", "consistency", "fluency", "relevance"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/v2/summeval"
RES = ROOT / "benchmarks/results_v2/summeval"
DATA.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
QS = P.task_spec(T).parsed  # dict qid -> Question

# Richer labeling descriptions (same option keys / levels as the task spec)
RICH = {
    "coherence": "Coherence (1-5 in the original study): the collective quality of all sentences. The summary should be well-structured and well-organized, building from sentence to sentence into a coherent body of information about the topic, not a heap of related facts. It should start by introducing the main subject (no dangling references like 'he' or 'the agency' without antecedent), proceed in a logical order, avoid repetition and contradictions, and end cleanly (a sentence cut off mid-way hurts).",
    "consistency": "Consistency: factual alignment between the summary and `source_document`. A consistent summary contains only statements entailed by the source document. Penalize hallucinated facts: details, names, numbers, dates, URLs, quotes or causal claims not in the source, and misattributions (a quote or action given to the wrong person). Copying sentences from the source is perfectly consistent.",
    "fluency": "Fluency: the quality of the individual sentences. Sentences should have no formatting problems, capitalization errors or obviously ungrammatical constructions (fragments, missing words, duplicated words, run-ons, sentences cut off mid-way) that make the text difficult to read.",
    "relevance": "Relevance: selection of important content from `source_document`. The summary should include only the important information (the central event and its key facts: who, what, where, outcome), with no redundancies (repeated facts) or excess, peripheral information (minor quotes, background color, photo captions).",
}

def rich_questions() -> dict[str, Question]:
    out = {}
    for d in DIMS:
        q = QS[d]
        out[d] = Question(type="score", instructions={"criterion": RICH[d], "question": "How well does `summary` meet `criterion`?"}, criteria=q.criteria)
    return out


def group_spearman(y, s, groups):
    by = defaultdict(list)
    for g, a, b in zip(groups, y, s):
        by[g].append((a, b))
    rh = []
    for pairs in by.values():
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            continue
        rh.append(stats.spearmanr(a, b).statistic)
    return float(np.mean(rh)) if rh else math.nan


def score_dev(pred: dict[str, np.ndarray], exs) -> dict:
    """pred: dim -> (N,) expected scores aligned with exs (dev examples with gold labels)."""
    g = [e.group for e in exs]
    out = {d: group_spearman([e.labels[d] for e in exs], pred[d], g) for d in DIMS if d in pred}
    out["mean"] = float(np.mean([out[d] for d in DIMS if d in out]))
    return out


def text_of(state: dict) -> str:
    """Encoder input: summary first, then the article (so right-truncation only cuts the article)."""
    return "summary: " + state["summary"].strip() + "\n\nsource_document: " + state["source_document"].strip()


import re as _re
_SPLIT = _re.compile(r"(?<=[.!?'\"])\s+(?=[A-Z0-9'\"`(])")
_W = _re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on at for with by from as is was were are be been it its this that he she they his her their him them who which said says has have had will would not but".split())


def _content(s):
    return {w for w in _W.findall(s.lower()) if w not in _STOP}


def text_align(state: dict, n_lead: int = 1, budget_chars: int = 700) -> str:
    """summary + the article sentences that best support it (lexical overlap), kept in article order, plus the lead
    sentence(s): lets a short encoder window see the evidence for consistency and the article's main point."""
    summ = state["summary"].strip()
    sents = [s.strip() for s in _SPLIT.split(state["source_document"].strip()) if s.strip()]
    sw = _content(summ)
    scored = []
    for i, s in enumerate(sents):
        cw = _content(s)
        scored.append((len(cw & sw) / (len(cw) ** 0.5 + 1), i))
    sents = [x[:300] for x in sents]
    order = list(range(min(n_lead, len(sents))))
    used = sum(len(sents[i]) for i in order)
    for sc, i in sorted(scored, key=lambda t: (-t[0], t[1])):
        if used >= budget_chars or sc <= 0:
            break
        if i not in order:
            order.append(i); used += len(sents[i])
    ctx = " ".join(sents[i] for i in order)
    return "summary: " + summ + "\n\nsource_document: " + ctx


def render(state: dict, ctx: str = "lead", chars: int = 700) -> str:
    return text_align(state, budget_chars=chars) if ctx == "align" else text_of(state)
