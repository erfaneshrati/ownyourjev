"""A trained decision model: encoder backbone + pooling + linear head + temperature, for one Question."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from ownyourjev.models.backbones import BackboneSpec, resolve_backbone
from ownyourjev.spec import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer


class DecisionModel(nn.Module):
    def __init__(
        self,
        question: Question,
        spec: BackboneSpec,
        backbone: PreTrainedModel,
        *,
        pooling: str | None = None,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.question = question
        self.spec = spec
        self.backbone = backbone
        self.pooling = pooling or spec.pooling
        self.max_len = max_len
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, question.num_outputs)
        self.register_buffer("temperature", torch.ones(1))
        self.calibrator = None  # optional post-hoc calibrator (ownyourjev.calibration.Calibrator)

    # ---- forward ----------------------------------------------------------------------------------
    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(input_ids, attention_mask)))

    def probs(self, logits: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        t = self.temperature.to(logits.device) if calibrated else 1.0
        z = logits.float() / t
        if self.question.type == "noul":
            return torch.sigmoid(z.squeeze(-1))
        return torch.softmax(z, dim=-1)

    # ---- loss -------------------------------------------------------------------------------------
    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor, *, label_smoothing: float = 0.0, ordinal_sigma: float = 0.0
    ) -> torch.Tensor:
        if self.question.type == "noul":
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), targets.float())
        if targets.dtype.is_floating_point and targets.ndim == 2:  # soft targets (distillation)
            return -(targets.float() * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
        if self.question.type == "score" and ordinal_sigma > 0:
            k = logits.shape[-1]
            idx = torch.arange(k, device=logits.device, dtype=torch.float32)
            soft = torch.exp(-((idx[None, :] - targets[:, None].float()) ** 2) / (2 * ordinal_sigma**2))
            soft = soft / soft.sum(-1, keepdim=True)
            return -(soft * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
        return F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

    # ---- answers ----------------------------------------------------------------------------------
    def answer(self, p: torch.Tensor | list[float] | float) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
        """Jev-style answer from the model's probabilities, after the post-hoc calibrator if one was fitted
        (`ownyourjev calibrate`; Choice/Score calibrators never change the chosen option)."""
        q = self.question
        cal = getattr(self, "calibrator", None)
        if cal is not None:
            arr = np.asarray(torch.as_tensor(p, dtype=torch.float64).reshape(-1))
            p = torch.as_tensor(cal.apply(arr[0:1] if q.type == "noul" else arr[None, :])[0])
        if q.type == "noul":
            v = float(p if isinstance(p, (int, float)) else torch.as_tensor(p).reshape(-1)[0])
            return NoulAnswer(noul=round(v, 6))
        probs = [float(x) for x in torch.as_tensor(p).reshape(-1)]
        if q.type == "choice":
            dist = {k: round(v, 6) for k, v in zip(q.options, probs)}
            best = max(dist, key=dist.get)
            return ChoiceAnswer(choice=best, probabilities=dist, confidence=round(dist[best], 6))
        dist = {str(i): round(v, 6) for i, v in enumerate(probs)}
        score = sum(i * v for i, v in enumerate(probs))
        return ScoreAnswer(score=round(score, 6), legend=q.legend, probabilities=dist, confidence=round(max(probs), 6))

    # ---- persistence --------------------------------------------------------------------------------
    def save(self, path: str | Path, tokenizer: AutoTokenizer | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        bb = self.backbone
        if hasattr(bb, "merge_and_unload"):
            bb = bb.merge_and_unload()
        bb.save_pretrained(path / "backbone")
        if tokenizer is not None:
            tokenizer.save_pretrained(path / "backbone")
        torch.save({"head": self.head.state_dict(), "temperature": self.temperature.detach().cpu()}, path / "head.pt")
        meta = {
            "question": self.question.model_dump(),
            "backbone": self.spec.name,
            "hf_id": self.spec.hf_id,
            "pooling": self.pooling,
            "max_len": self.max_len,
            "spec_hash": self.question.spec_hash(),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> "DecisionModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        spec = resolve_backbone(meta["backbone"])
        backbone = AutoModel.from_pretrained(path / "backbone", dtype=dtype or torch.float32)
        m = cls(Question(**meta["question"]), spec, backbone, pooling=meta["pooling"], max_len=meta["max_len"])
        st = torch.load(path / "head.pt", map_location="cpu")
        m.head.load_state_dict(st["head"])
        m.temperature.copy_(st["temperature"])
        m.calibrator = load_calibrator(path)
        return m.to(device).eval()


def load_calibrator(path: str | Path):
    """The post-hoc calibrator saved by `ownyourjev calibrate` (calibration.json), or None."""
    from ownyourjev.calibration import Calibrator

    f = Path(path) / "calibration.json"
    return Calibrator.from_dict(json.loads(f.read_text())) if f.is_file() else None


def load_backbone(spec: BackboneSpec, dtype: torch.dtype = torch.float32) -> tuple[PreTrainedModel, Any]:
    """fp32 master weights by default: some checkpoints (e.g. gte-modernbert) are stored in fp16 and
    transformers>=5 keeps the stored dtype, which makes AdamW fine-tuning collapse."""
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModel.from_pretrained(spec.hf_id, dtype=dtype)
    return model, tok


def expected_score(probs: torch.Tensor) -> torch.Tensor:
    k = probs.shape[-1]
    return (probs * torch.arange(k, device=probs.device, dtype=probs.dtype)).sum(-1)


def is_finite(x: float) -> bool:
    return x is not None and math.isfinite(x)
