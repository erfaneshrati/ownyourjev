"""Synthetic (state, labels) generation from a question spec alone.

For users with no data at all. Batches of examples are requested per target answer so the resulting set is
balanced over options / levels, with topic and style hints rotated for diversity; near-duplicates are removed
with MinHash.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Any

from datasketch import MinHash, MinHashLSH

from ownyourjev.data import Example
from ownyourjev.llm.client import LLMClient
from ownyourjev.spec import Question, canonical_json, parse_label, render_state

STYLE_HINTS = [
    "short and casual",
    "long and detailed",
    "formal",
    "with typos and informal grammar",
    "terse, fragment-like",
    "emotional",
    "neutral and factual",
    "with numbers, names or dates",
    "ambiguous or borderline",
    "clearly obvious",
]


def _targets(questions: dict[str, Question]) -> list[dict[str, Any]]:
    """Cycle through target answers for the first question; other questions are left to the model."""
    qid, q = next(iter(questions.items()))
    if q.type == "choice":
        return [{qid: k} for k in q.options]
    if q.type == "score":
        return [{qid: i} for i in range(len(q.criteria))]  # type: ignore[arg-type]
    return [{qid: True}, {qid: False}]


def build_synth_messages(
    questions: dict[str, Question],
    *,
    n: int,
    target: dict[str, Any],
    state_fields: list[str] | None,
    domain: str,
    style: str,
    seeds: list[Example] | None,
) -> list[dict]:
    sys_parts = [
        "You generate realistic, diverse training data for a text classifier. Reply with a single JSON object only.",
        "",
        "## Questions the classifier must answer about each STATE",
    ]
    for qid, q in questions.items():
        sys_parts.append(f"### {qid}\n{q.describe()}\n")
    if state_fields:
        shape = "{" + ", ".join(f'"{f}": "<text>"' for f in state_fields) + "}"
        sys_parts.append(f"Each STATE is an object with fields: {shape}")
    else:
        sys_parts.append("Each STATE is a plain text string.")
    if seeds:
        sys_parts.append("\n## Real examples (match their domain and distribution; never copy them)")
        for ex in seeds[:20]:
            sys_parts.append("STATE: " + canonical_json(ex.state) + "\nANSWER: " + canonical_json(ex.labels or {}))
    user = (
        f"Domain: {domain}\nStyle for this batch: {style}\n"
        f"Generate {n} distinct STATE examples whose correct answer is {canonical_json(target)}. "
        "Vary topics, wording, length and difficulty; include hard and borderline cases that are still correctly labeled. "
        'Return: {"examples": [{"state": <STATE>, "labels": {<question id>: <answer>, ...}}, ...]} '
        "with answers for every question."
    )
    return [{"role": "system", "content": "\n".join(sys_parts)}, {"role": "user", "content": user}]


def _shingles(text: str, k: int = 3) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i : i + k]) for i in range(max(1, len(toks) - k + 1))}


class Deduper:
    def __init__(self, threshold: float = 0.8, num_perm: int = 64):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.n = 0
        self.exact: set[str] = set()

    def add(self, text: str) -> bool:
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        if norm in self.exact:
            return False
        m = MinHash(num_perm=self.num_perm)
        for s in _shingles(text):
            m.update(s.encode())
        if self.lsh.query(m):
            return False
        self.exact.add(norm)
        self.lsh.insert(str(self.n), m)
        self.n += 1
        return True


async def synthesize(
    client: LLMClient,
    questions: dict[str, Question],
    n_total: int,
    *,
    purpose: str,
    domain: str = "general",
    state_fields: list[str] | None = None,
    seeds: list[Example] | None = None,
    batch_size: int = 8,
    seed: int = 0,
    progress: bool = True,
) -> list[Example]:
    rng = random.Random(seed)
    targets = _targets(questions)
    n_batches = -(-n_total // batch_size)
    jobs = []
    for b, target in zip(range(n_batches), itertools.cycle(targets)):
        style = STYLE_HINTS[(b * 7 + rng.randrange(len(STYLE_HINTS))) % len(STYLE_HINTS)]
        jobs.append((b, target, style))

    async def one(job: tuple[int, dict, str]) -> list[dict]:
        b, target, style = job
        msgs = build_synth_messages(
            questions, n=batch_size, target=target, state_fields=state_fields, domain=domain, style=style, seeds=seeds
        )
        try:
            raw = await client.complete_json(msgs, purpose=purpose, max_tokens=200 * batch_size + 200)
        except (RuntimeError, ValueError):
            return []
        items = raw.get("examples") if isinstance(raw, dict) else raw
        return items if isinstance(items, list) else []

    batches = await client.map(jobs, one, desc=f"synth:{purpose}", progress=progress)
    dedupe = Deduper()
    out: list[Example] = []
    for b, items in enumerate(batches):
        for j, it in enumerate(items):
            if not isinstance(it, dict) or "state" not in it:
                continue
            state = it["state"]
            if state_fields and not (isinstance(state, dict) and all(f in state for f in state_fields)):
                continue
            if not state_fields and not isinstance(state, str):
                continue
            try:
                labels = {qid: parse_label(q, (it.get("labels") or {})[qid]) for qid, q in questions.items()}
            except (KeyError, ValueError):
                continue
            if not dedupe.add(render_state(state)):
                continue
            out.append(Example(f"synth-{b}-{j}", state, labels, None, "", {"synthetic": True, "labeler": client.model}))
    return out[:n_total]
