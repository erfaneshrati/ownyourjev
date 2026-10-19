"""Jev-compatible local HTTP server.

    POST /v1/systemone      {"state": ..., "questions": {qid: {"type", "instructions", "criteria"}}}
    -> {"answers": {qid: ...}, "model": "ownyourjev", "usage": {...}}

Models are looked up in a registry directory: every sub-directory is a saved DecisionModel; a question is
matched to a model by its ``spec_hash`` (type + instructions + criteria) or, as a fallback, by the
question id given at training time. Unknown questions return HTTP 404 with the list of known specs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from ownyourjev.models.decision import DecisionModel
from ownyourjev.spec import DecisionRequest, DecisionResponse, Question, Usage, render_state
from ownyourjev.train import pick_device


class Registry:
    def __init__(self, root: str | Path, device: str | None = None):
        self.root = Path(root)
        self.device = device or pick_device(min_free_gb=1.0)
        self.by_hash: dict[str, tuple[DecisionModel, Any]] = {}
        self.by_name: dict[str, tuple[DecisionModel, Any]] = {}
        for d in sorted(p for p in self.root.iterdir() if (p / "meta.json").is_file()):
            m = DecisionModel.load(d, device=self.device, dtype=torch.bfloat16 if self.device.startswith("cuda") else None)
            tok = AutoTokenizer.from_pretrained(d / "backbone")
            self.by_hash[m.question.spec_hash()] = (m, tok)
            self.by_name[d.name] = (m, tok)

    def resolve(self, qid: str, q: Question) -> tuple[DecisionModel, Any]:
        hit = self.by_hash.get(q.spec_hash()) or self.by_name.get(qid)
        if hit is None:
            raise HTTPException(404, detail={"error": f"no trained model for question {qid!r}", "known": list(self.by_name)})
        return hit

    @torch.no_grad()
    def answer(self, qid: str, q: Question, state: Any) -> tuple[Any, int]:
        model, tok = self.resolve(qid, q)
        enc = tok([render_state(state)], truncation=True, max_length=model.max_len, return_tensors="pt")
        logits = model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        p = model.probs(logits)[0].cpu()
        return model.answer(p), int(enc["input_ids"].shape[1])


def create_app(registry_dir: str | Path, device: str | None = None) -> FastAPI:
    app = FastAPI(title="ownyourjev", version="0.1.0")
    reg = Registry(registry_dir, device)

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": name, "question": m.question.model_dump(), "spec_hash": m.question.spec_hash()} for name, (m, _) in reg.by_name.items()]}

    @app.post("/v1/systemone", response_model=DecisionResponse)
    @app.post("/v1/decisions", response_model=DecisionResponse)
    def systemone(req: DecisionRequest) -> DecisionResponse:
        t0 = time.time()
        answers = {}
        tokens = 0
        for qid, q in req.questions.items():
            ans, n = reg.answer(qid, q, req.state)
            answers[qid] = ans
            tokens += n
        return DecisionResponse(answers=answers, usage=Usage(input_tokens=tokens, output_tokens=0, cost=0.0, latency_ms=(time.time() - t0) * 1000))

    return app
