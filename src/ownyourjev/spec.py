"""Jev-compatible question / request / answer schema.

The request and response shapes mirror TypeSafe's System One API so a trained model can be served as a
drop-in replacement:

    request  = {"state": <str | dict | list>, "questions": {qid: {"type", "instructions", "criteria"}}}
    response = {"answers": {qid: <ChoiceAnswer | ScoreAnswer | NoulAnswer>}, "model": ..., "usage": ...}

- choice: ``criteria`` is a mapping option_key -> description (description may be null).
- score:  ``criteria`` is an ordered list of level descriptions; the answer is the probability-weighted
          mean of the level indices (0 .. n-1), exactly like Jev.
- noul:   ``criteria`` is optional ``{"true": ..., "false": ...}``; the answer is P(yes).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["choice", "score", "noul"]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Question(BaseModel):
    type: QuestionType
    instructions: Any
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice questions need a criteria mapping with at least 2 options")
            if len(self.criteria) > 255:
                raise ValueError("choice questions support at most 255 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score questions need an ordered list of 2 to 10 levels")
        elif self.type == "noul":
            if self.criteria is not None and (
                not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}
            ):
                raise ValueError('noul criteria must be null or a mapping with keys "true"/"false"')
        return self

    # ---- derived -------------------------------------------------------------------------------
    @property
    def options(self) -> list[str]:
        """Output labels in a fixed order: option keys, level indices as str, or ["true"] for noul."""
        if self.type == "choice":
            return list(self.criteria.keys())  # type: ignore[union-attr]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]  # type: ignore[arg-type]
        return ["true"]

    @property
    def num_outputs(self) -> int:
        return 1 if self.type == "noul" else len(self.options)

    @property
    def legend(self) -> dict[str, Any]:
        if self.type != "score":
            return {}
        return {str(i): lvl for i, lvl in enumerate(self.criteria)}  # type: ignore[arg-type]

    def spec_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump()).encode()).hexdigest()[:16]

    def describe(self) -> str:
        """Human-readable rendering of the question used in LLM prompts."""
        ins = self.instructions if isinstance(self.instructions, str) else canonical_json(self.instructions)
        lines = [f"Type: {self.type}", f"Question: {ins}"]
        if self.type == "choice":
            lines.append("Options (answer with the option key exactly):")
            for k, v in self.criteria.items():  # type: ignore[union-attr]
                lines.append(f"  - {k}" + (f": {v}" if v else ""))
        elif self.type == "score":
            lines.append("Ordered levels (answer with the level index as an integer):")
            for i, v in enumerate(self.criteria):  # type: ignore[arg-type]
                lines.append(f"  {i}: {v if isinstance(v, str) else canonical_json(v)}")
        else:
            lines.append("Answer with true or false.")
            if self.criteria:
                for k in ("true", "false"):
                    if k in self.criteria:
                        lines.append(f"  {k}: {self.criteria[k]}")
        return "\n".join(lines)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, Question]
    model: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float | None = None


class DecisionResponse(BaseModel):
    answers: dict[str, Answer]
    model: str = "ownyourjev"
    usage: Usage = Field(default_factory=Usage)


# ---- state rendering ------------------------------------------------------------------------------


def render_state(state: Any) -> str:
    """Render a Jev ``state`` (str, dict or list) as plain text.

    The same rendering feeds both the labeling LLM and the encoder, so what the encoder learns to read
    is exactly what the labeler read. Dict keys are written as ``key: value`` blocks; the question
    instructions refer to them in backticks, e.g. "Do `sentence_1` and `sentence_2` mean the same?".
    """
    if state is None:
        return ""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        parts = []
        for k, v in state.items():
            vs = v.strip() if isinstance(v, str) else canonical_json(v)
            parts.append(f"{k}: {vs}" if "\n" not in vs else f"{k}:\n{vs}")
        return "\n\n".join(parts)
    if isinstance(state, list):
        return "\n\n".join(render_state(s) for s in state)
    return str(state)


def parse_label(q: Question, raw: Any) -> Any:
    """Normalize a raw LLM / gold label into the canonical form for ``q``.

    choice -> option key (str); score -> int level index; noul -> bool.  Raises ValueError if invalid.
    """
    if q.type == "choice":
        keys = list(q.criteria.keys())  # type: ignore[union-attr]
        if isinstance(raw, dict) and "choice" in raw:
            raw = raw["choice"]
        s = str(raw).strip()
        if s in keys:
            return s
        low = {k.lower(): k for k in keys}
        if s.lower() in low:
            return low[s.lower()]
        s2 = s.strip("`'\" .")
        if s2.lower() in low:
            return low[s2.lower()]
        raise ValueError(f"choice {raw!r} not in options")
    if q.type == "score":
        n = len(q.criteria)  # type: ignore[arg-type]
        if isinstance(raw, dict) and "score" in raw:
            raw = raw["score"]
        if isinstance(raw, bool):
            raise ValueError("score must be an integer level")
        try:
            v = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"score {raw!r} not numeric") from e
        iv = int(round(v))
        if not 0 <= iv < n:
            raise ValueError(f"score {raw!r} out of range 0..{n - 1}")
        return iv
    # noul
    if isinstance(raw, dict) and "noul" in raw:
        raw = raw["noul"]
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw >= 0.5)
    s = str(raw).strip().lower().strip("`'\" .")
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"noul {raw!r} not boolean")


def label_to_index(q: Question, label: Any) -> int:
    """Canonical label -> class index used by the heads (noul: 1 for true)."""
    if q.type == "choice":
        return q.options.index(label)
    if q.type == "score":
        return int(label)
    return int(bool(label))
