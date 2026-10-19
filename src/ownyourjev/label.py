"""LLM labeling of unlabeled states against a Jev-style question spec.

The prompt is built from the *same* ``Question`` objects the encoder is trained for, so the labeler and
the trained model answer literally the same question. The question block (and optional few-shot seed
examples) go in the system message, which keeps a stable prefix for provider-side prompt caching; the
state goes in the user message.
"""

from __future__ import annotations

import json
from typing import Any

from ownyourjev.data import Example
from ownyourjev.llm.client import LLMClient
from ownyourjev.spec import Question, canonical_json, parse_label, render_state

SYSTEM_PREAMBLE = (
    "You are a meticulous, consistent annotator. You read a STATE and answer one or more typed questions "
    "about it. Follow the option/level descriptions literally. Reply with a single JSON object and nothing else."
)


def output_format_line(questions: dict[str, Question], distribution: bool = False) -> str:
    if distribution:
        return output_format_distribution(questions)
    parts = []
    for qid, q in questions.items():
        if q.type == "choice":
            parts.append(f'"{qid}": "<option key>"')
        elif q.type == "score":
            parts.append(f'"{qid}": <integer level index 0-{len(q.criteria) - 1}>')  # type: ignore[arg-type]
        else:
            parts.append(f'"{qid}": <true|false>')
    return "{" + ", ".join(parts) + "}"


def output_format_distribution(questions: dict[str, Question]) -> str:
    parts = []
    for qid, q in questions.items():
        if q.type == "noul":
            parts.append(f'"{qid}": <probability between 0 and 1 that the answer is true>')
        else:
            keys = q.options
            shown = ", ".join(f'"{k}": <p>' for k in keys[:6]) + (", ..." if len(keys) > 6 else "")
            parts.append(f'"{qid}": {{{shown}}} (a probability for EVERY option, summing to 1)')
    return "{" + ", ".join(parts) + "}"


def parse_distribution(q: Question, raw) -> tuple[object, list[float] | float]:
    """-> (canonical hard label, soft target). noul: soft = P(true); choice/score: list over q.options."""
    if q.type == "noul":
        if isinstance(raw, dict):
            raw = raw.get("true", raw.get("p", raw.get("noul")))
        p = float(raw)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p} out of range")
        return p >= 0.5, p
    if not isinstance(raw, dict):
        raise ValueError("expected a mapping option -> probability")
    probs = []
    norm = {str(k).strip().lower(): v for k, v in raw.items()}
    for k in q.options:
        v = raw.get(k, norm.get(k.lower(), 0.0))
        probs.append(max(0.0, float(v)))
    tot = sum(probs)
    if tot <= 0:
        raise ValueError("all-zero distribution")
    probs = [x / tot for x in probs]
    best = max(range(len(probs)), key=probs.__getitem__)
    label = q.options[best] if q.type == "choice" else best
    return label, probs


def seeds_block(questions: dict[str, Question], seeds: list[Example]) -> str:
    lines = ["## Examples of correct answers"]
    for ex in seeds:
        labels = ex.labels or {}
        lines.append("STATE:\n" + render_state(ex.state))
        lines.append("ANSWER: " + canonical_json({qid: labels[qid] for qid in questions if qid in labels}))
        lines.append("")
    return "\n".join(lines)


def build_messages(
    questions: dict[str, Question],
    state: Any,
    *,
    seeds: list[Example] | None = None,
    error: str | None = None,
    guidance: str | None = None,
    distribution: bool = False,
    system_preamble: str | None = None,
) -> list[dict]:
    """``guidance``: extra task-specific labeling instructions (prompt optimisation); ``distribution``: ask for a
    probability per option instead of one answer (soft labels)."""
    sys_parts = [system_preamble or SYSTEM_PREAMBLE, "", "## Questions"]
    for qid, q in questions.items():
        sys_parts.append(f"### {qid}\n{q.describe()}\n")
    if guidance:
        sys_parts.append("## Labeling guidelines\n" + guidance.strip() + "\n")
    sys_parts.append("## Output format\nReturn exactly this JSON object: " + output_format_line(questions, distribution))
    if seeds:
        sys_parts.append("")
        sys_parts.append(seeds_block(questions, seeds))
    user = "## STATE\n" + render_state(state) + "\n\n## ANSWER (JSON only)"
    if error:
        user += f"\n\nYour previous answer was invalid: {error}. Answer again following the output format exactly."
    return [{"role": "system", "content": "\n".join(sys_parts)}, {"role": "user", "content": user}]


def parse_labels(questions: dict[str, Question], raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("reply is not a JSON object")
    # tolerate a single-question reply given as a bare value under a different key
    out: dict[str, Any] = {}
    for qid, q in questions.items():
        if qid in raw:
            out[qid] = parse_label(q, raw[qid])
        elif len(questions) == 1 and len(raw) == 1:
            out[qid] = parse_label(q, next(iter(raw.values())))
        else:
            raise ValueError(f"missing answer for {qid!r}")
    return out


async def label_examples(
    client: LLMClient,
    questions: dict[str, Question],
    examples: list[Example],
    *,
    purpose: str,
    seeds: list[Example] | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
    progress: bool = True,
    guidance: str | None = None,
    distribution: bool = False,
    system_preamble: str | None = None,
) -> list[Example]:
    """Return copies of ``examples`` with ``labels`` filled (None where the LLM failed twice).
    With ``distribution`` the soft targets are stored in ``meta["soft"][qid]``."""
    max_tokens = max_tokens or ((64 + 24 * len(questions)) if not distribution else (96 + sum(12 * q.num_outputs for q in questions.values())))

    async def one(ex: Example) -> Example:
        error = None
        for _ in range(retries + 1):
            msgs = build_messages(questions, ex.state, seeds=seeds, error=error, guidance=guidance, distribution=distribution, system_preamble=system_preamble)
            try:
                raw = await client.complete_json(msgs, purpose=purpose, max_tokens=max_tokens)
                if distribution:
                    if not isinstance(raw, dict):
                        raise ValueError("reply is not a JSON object")
                    labels, soft = {}, {}
                    for qid, q in questions.items():
                        if qid not in raw:
                            raise ValueError(f"missing answer for {qid!r}")
                        labels[qid], soft[qid] = parse_distribution(q, raw[qid])
                    return Example(ex.uid, ex.state, labels, ex.gold, ex.group, {**ex.meta, "labeler": client.model, "soft": soft})
                labels = parse_labels(questions, raw)
                return Example(ex.uid, ex.state, labels, ex.gold, ex.group, {**ex.meta, "labeler": client.model})
            except (ValueError, json.JSONDecodeError) as e:
                error = str(e)[:200]
            except RuntimeError as e:  # transport failure after retries: keep going, mark invalid
                error = f"transport: {e}"[:200]
                break
        return Example(ex.uid, ex.state, None, ex.gold, ex.group, {**ex.meta, "label_error": error})

    return await client.map(examples, one, desc=f"label:{purpose}", progress=progress)
