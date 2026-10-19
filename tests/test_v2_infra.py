import asyncio
import json

import pytest
import torch

from ownyourjev.data import Example
from ownyourjev.label import build_messages, label_examples, parse_distribution
from ownyourjev.llm import Budget, BudgetExceeded, FakeLLMClient
from ownyourjev.spec import Question


def test_parse_distribution_and_soft_labeling():
    qc = Question(type="choice", instructions="i", criteria={"a": None, "b": None, "c": None})
    lab, soft = parse_distribution(qc, {"a": 0.2, "B": 0.6, "c": 0.4})
    assert lab == "b" and abs(sum(soft) - 1) < 1e-9 and soft[1] == pytest.approx(0.5)
    qn = Question(type="noul", instructions="i")
    assert parse_distribution(qn, 0.8) == (True, 0.8)
    qs = Question(type="score", instructions="i", criteria=["l", "m", "h"])
    assert parse_distribution(qs, {"0": 0.1, "1": 0.2, "2": 0.7})[0] == 2

    client = FakeLLMClient(lambda m: json.dumps({"x": {"a": 0.1, "b": 0.1, "c": 0.8}}))
    out = asyncio.run(label_examples(client, {"x": qc}, [Example("1", "t")], purpose="t", distribution=True, guidance="Prefer c.", progress=False))
    assert out[0].labels == {"x": "c"} and out[0].meta["soft"]["x"][2] == pytest.approx(0.8)
    sys_msg = client.calls[0][0]["content"]
    assert "Labeling guidelines" in sys_msg and "Prefer c." in sys_msg and "probability for EVERY option" in sys_msg


def test_budget_scope_cap(tmp_path):
    b = Budget(tmp_path / "b.db", cap_usd=10, scope="exp1", scope_cap_usd=0.05)
    b.record(purpose="label", model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.04)
    assert b.summary()[0].purpose == "exp1:label"
    b.check(0.005)
    with pytest.raises(BudgetExceeded):
        b.check(0.02)
    other = Budget(tmp_path / "b.db", cap_usd=10, scope="exp2", scope_cap_usd=0.05)
    other.check(0.04)  # separate scope, same global ledger
    assert other.spent() == pytest.approx(0.04)


def test_soft_loss():
    from transformers import BertConfig, BertModel

    from ownyourjev.models.backbones import BackboneSpec
    from ownyourjev.models.decision import DecisionModel

    bb = BertModel(BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=50))
    q = Question(type="choice", instructions="i", criteria={"a": None, "b": None})
    m = DecisionModel(q, BackboneSpec("t", "t", "mean", 1e-3, 1e-3, 16), bb, max_len=16)
    logits = torch.zeros(2, 2)
    soft = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    hard = torch.tensor([0, 0])
    assert torch.isclose(m.loss(logits, soft), m.loss(logits, hard))  # uniform logits: same CE


def test_gpu_slot_cpu_fallback(monkeypatch):
    import ownyourjev.gpu as g

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with g.gpu_slot(1.0) as dev:
        assert dev == "cpu"


def test_build_messages_custom_preamble():
    q = {"x": Question(type="noul", instructions="i")}
    msgs = build_messages(q, "s", system_preamble="You are a strict annotator.")
    assert msgs[0]["content"].startswith("You are a strict annotator.")
