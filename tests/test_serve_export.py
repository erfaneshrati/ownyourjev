"""Server + ONNX export round trip with a tiny random encoder (no downloads, CPU only)."""


import numpy as np
from fastapi.testclient import TestClient
from transformers import BertConfig, BertModel, BertTokenizerFast

from ownyourjev.models.backbones import BACKBONES, BackboneSpec
from ownyourjev.models.decision import DecisionModel
from ownyourjev.spec import Question


def _tiny(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(95)] + ["hello", "world", "spam"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"))
    bb = BertModel(BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=len(vocab), max_position_embeddings=64))
    spec = BackboneSpec("tiny", "tiny", "mean", 1e-3, 1e-3, 32)
    BACKBONES["tiny"] = spec  # so DecisionModel.load can resolve it
    return tok, bb, spec


def test_serve_and_onnx(tmp_path):
    tok, bb, spec = _tiny(tmp_path)
    reg = tmp_path / "models"
    qs = {
        "team": Question(type="choice", instructions="Which team?", criteria={"billing": None, "tech": None}),
        "urgency": Question(type="score", instructions="How urgent?", criteria=["low", "mid", "high"]),
        "churn": Question(type="noul", instructions="Will they churn?"),
    }
    for qid, q in qs.items():
        m = DecisionModel(q, spec, bb, max_len=32)
        m.save(reg / qid, tok)
    from ownyourjev.serve import create_app

    app = create_app(reg, device="cpu")
    c = TestClient(app)
    req = {"state": {"ticket": "hello world spam"}, "questions": {k: v.model_dump() for k, v in qs.items()}}
    r = c.post("/v1/systemone", json=req)
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]
    assert ans["team"]["choice"] in ("billing", "tech") and abs(sum(ans["team"]["probabilities"].values()) - 1) < 1e-4
    assert 0 <= ans["urgency"]["score"] <= 2 and ans["urgency"]["legend"]["2"] == "high"
    assert 0 <= ans["churn"]["noul"] <= 1
    # unknown question -> 404 with the list of known models
    bad = dict(req, questions={"x": {"type": "noul", "instructions": "unknown?"}})
    assert c.post("/v1/systemone", json=bad).status_code == 404
    # same answers through ONNX
    from ownyourjev.export import OnnxPredictor, export_onnx

    p = export_onnx(reg / "team")
    pred = OnnxPredictor(reg / "team", p)
    onnx_probs = pred.probs([req["state"]])[0]
    torch_probs = np.array([ans["team"]["probabilities"]["billing"], ans["team"]["probabilities"]["tech"]])
    assert np.allclose(onnx_probs, torch_probs, atol=1e-3)
