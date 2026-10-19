"""Shared-encoder, two-head model for toxigen (one forward pass answers both `toxic` and `toxicity`).

Output layout (logits and probs): column 0 = toxic (sigmoid -> P(true)); columns 1..5 = toxicity levels
(softmax). Saved in a DecisionModel-like directory (backbone/, head.pt, meta.json) so that
ownyourjev.export.OnnxPredictor and benchmarks.v2.latency can load the ONNX export unchanged
(meta.question is the toxicity question; the 'probs' output has 6 columns).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ownyourjev.models.backbones import resolve_backbone


class MultiHead(nn.Module):
    def __init__(self, backbone, hidden: int, pooling: str = "cls", dropout: float = 0.1, max_len: int = 64, backbone_name: str = "gte-modernbert"):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.max_len = max_len
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 6)
        self.register_buffer("temperature", torch.ones(2))  # [toxic, toxicity]

    def embed(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            return h[:, 0]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, mask):
        return self.head(self.dropout(self.embed(ids, mask)))

    def probs(self, logits, calibrated: bool = True):
        z = logits.float()
        t0, t1 = (self.temperature[0], self.temperature[1]) if calibrated else (1.0, 1.0)
        return torch.cat([torch.sigmoid(z[:, :1] / t0), torch.softmax(z[:, 1:] / t1, -1)], -1)

    @staticmethod
    def loss(logits, y_tox, y_dist, w_tox: float = 1.0):
        z = logits.float()
        l1 = F.binary_cross_entropy_with_logits(z[:, 0], y_tox.float())
        l2 = -(y_dist.float() * F.log_softmax(z[:, 1:], -1)).sum(-1).mean()
        return w_tox * l1 + l2

    def save(self, path, tok, question_dump: dict):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone"); tok.save_pretrained(path / "backbone")
        torch.save({"head": self.head.state_dict(), "temperature": self.temperature.detach().cpu()}, path / "head.pt")
        (path / "meta.json").write_text(json.dumps({"question": question_dump, "backbone": self.backbone_name, "pooling": self.pooling,
                                                    "max_len": self.max_len, "multihead": ["toxic", "toxicity"]}, indent=1))

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path); meta = json.loads((path / "meta.json").read_text())
        bb = AutoModel.from_pretrained(path / "backbone", dtype=torch.float32)
        m = cls(bb, bb.config.hidden_size, meta["pooling"], 0.1, meta["max_len"], meta["backbone"])
        st = torch.load(path / "head.pt", map_location="cpu")
        m.head.load_state_dict(st["head"]); m.temperature.copy_(st["temperature"])
        return m.to(device).eval(), AutoTokenizer.from_pretrained(path / "backbone")


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m.probs(self.m(input_ids, attention_mask))


def export_onnx(model_dir, opset: int = 17) -> Path:
    m, tok = MultiHead.load(model_dir)
    m.eval()
    enc = tok(["example input for tracing", "second"], padding=True, return_tensors="pt")
    out = Path(model_dir) / "model.onnx"
    torch.onnx.export(_Wrap(m), (enc["input_ids"], enc["attention_mask"]), str(out), input_names=["input_ids", "attention_mask"],
                      output_names=["probs"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "probs": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return out


def onnx_probs(model_dir, texts: list[str], onnx_name: str = "model.int8.onnx", threads: int = 8, batch: int = 64) -> np.ndarray:
    import onnxruntime as ort
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "backbone")
    o = ort.SessionOptions(); o.intra_op_num_threads = threads
    s = ort.InferenceSession(str(Path(model_dir) / onnx_name), o, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=meta["max_len"], return_tensors="np")
        out.append(s.run(["probs"], {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
    return np.concatenate(out)
