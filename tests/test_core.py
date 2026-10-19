import asyncio
import json

import numpy as np
import pytest
import torch

from ownyourjev.data import Example, kfold, stratified_subsample, train_val_split
from ownyourjev.evaluate import binary_metrics, choice_metrics, ece, score_metrics
from ownyourjev.label import build_messages, label_examples, parse_labels
from ownyourjev.llm import Budget, BudgetExceeded, FakeLLMClient
from ownyourjev.spec import DecisionRequest, Question, parse_label, render_state


def test_question_validation():
    Question(type="choice", instructions="x", criteria={"a": None, "b": "desc"})
    Question(type="score", instructions="x", criteria=["lo", "hi"])
    Question(type="noul", instructions="x")
    Question(type="noul", instructions="x", criteria={"true": "yes", "false": "no"})
    with pytest.raises(ValueError):
        Question(type="choice", instructions="x", criteria={"only": None})
    with pytest.raises(ValueError):
        Question(type="score", instructions="x", criteria=["one"])
    with pytest.raises(ValueError):
        Question(type="noul", instructions="x", criteria={"maybe": "?"})


def test_spec_hash_stable_and_order_sensitive():
    a = Question(type="choice", instructions="q", criteria={"x": None, "y": None})
    b = Question(type="choice", instructions="q", criteria={"x": None, "y": None})
    c = Question(type="choice", instructions="q!", criteria={"x": None, "y": None})
    assert a.spec_hash() == b.spec_hash() != c.spec_hash()


def test_parse_label():
    qc = Question(type="choice", instructions="q", criteria={"Sci/Tech": None, "World": None})
    assert parse_label(qc, " world ") == "World"
    assert parse_label(qc, {"choice": "Sci/Tech"}) == "Sci/Tech"
    with pytest.raises(ValueError):
        parse_label(qc, "Sports")
    qs = Question(type="score", instructions="q", criteria=["a", "b", "c"])
    assert parse_label(qs, "2") == 2 and parse_label(qs, 1.2) == 1
    with pytest.raises(ValueError):
        parse_label(qs, 3)
    qn = Question(type="noul", instructions="q")
    assert parse_label(qn, "yes") is True and parse_label(qn, False) is False and parse_label(qn, 0.7) is True


def test_render_state():
    assert render_state("  hi ") == "hi"
    assert render_state({"a": "x", "b": "y"}) == "a: x\n\nb: y"
    assert "passage:\nline1\nline2" in render_state({"passage": "line1\nline2"})


def test_label_examples_with_fake_llm_and_retry():
    qs = {"spam": Question(type="noul", instructions="Is `message` spam?"), "tone": Question(type="score", instructions="urgency", criteria=["low", "high"])}
    calls = {"n": 0}

    def responder(messages):
        calls["n"] += 1
        state = messages[-1]["content"]
        if "retry-me" in state and "previous answer was invalid" not in state:
            return "garbage"
        spam = "WIN" in state
        return json.dumps({"spam": spam, "tone": 1 if spam else 0})

    client = FakeLLMClient(responder)
    exs = [Example("1", {"message": "WIN a prize"}), Example("2", {"message": "lunch?"}), Example("3", {"message": "retry-me WIN"})]
    out = asyncio.run(label_examples(client, qs, exs, purpose="t", progress=False))
    assert out[0].labels == {"spam": True, "tone": 1}
    assert out[1].labels == {"spam": False, "tone": 0}
    assert out[2].labels == {"spam": True, "tone": 1}  # succeeded on retry
    assert calls["n"] == 4
    msgs = build_messages(qs, exs[0].state, seeds=[out[1]])
    assert "Examples of correct answers" in msgs[0]["content"] and "lunch?" in msgs[0]["content"]


def test_parse_labels_single_question_tolerance():
    qs = {"answer": Question(type="noul", instructions="q")}
    assert parse_labels(qs, {"result": "yes"}) == {"answer": True}
    with pytest.raises(ValueError):
        parse_labels(qs, {"a": True, "b": False})


def test_budget_hard_stop(tmp_path):
    b = Budget(tmp_path / "b.db", cap_usd=0.01)
    b.record(purpose="x", model="m", prompt_tokens=10, completion_tokens=1, cost_usd=0.009)
    b.check(0.0005)
    with pytest.raises(BudgetExceeded):
        b.check(0.002)
    b.record(purpose="x", model="m", prompt_tokens=10, completion_tokens=1, cost_usd=0.002)
    with pytest.raises(BudgetExceeded):
        b.check(0.0)
    assert b.summary()[0].calls == 2


def test_metrics_known_values():
    assert ece(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    m = binary_metrics([True, False, True, False], np.array([0.9, 0.1, 0.4, 0.6]))
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    c = choice_metrics(["a", "b"], np.array([[0.8, 0.2], [0.3, 0.7]]), ["a", "b"])
    assert c["accuracy"] == 1.0
    s = score_metrics([0, 1, 2], np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0.2, 0.8]]), groups=["g", "g", "g"])
    assert s["spearman"] == pytest.approx(1.0) and s["argmax_accuracy"] == 1.0 and s["group_spearman"] == pytest.approx(1.0)


def test_data_helpers():
    exs = [Example(str(i), f"t{i}", {"a": i % 3}, group=str(i % 4)) for i in range(40)]
    sub = stratified_subsample(exs, 9, key=lambda e: e.labels["a"], seed=1)
    assert len(sub) == 9 and sorted({e.labels["a"] for e in sub}) == [0, 1, 2]
    tr, va = train_val_split(exs, 0.1, seed=0)
    assert len(tr) + len(va) == 40 and len(va) >= 4
    folds = kfold(exs, 4, grouped=True)
    for tr_i, held_i in folds:
        assert {exs[i].group for i in tr_i}.isdisjoint({exs[i].group for i in held_i})


def test_decision_request_roundtrip():
    req = DecisionRequest(state={"text": "x"}, questions={"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}})
    assert req.questions["q"].options == ["a", "b"]


def test_decision_model_heads_cpu():
    """Tiny randomly-initialised backbone: shapes, losses, answers and save/load, no download."""

    from ownyourjev.models.backbones import BackboneSpec
    from ownyourjev.models.decision import DecisionModel

    from transformers import BertConfig, BertModel

    bb = BertModel(BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=100))
    spec = BackboneSpec("tiny", "tiny", "mean", 1e-3, 1e-3, 32)
    ids = torch.randint(0, 100, (3, 7))
    mask = torch.ones_like(ids)
    for q, y in [
        (Question(type="choice", instructions="i", criteria={"a": None, "b": None, "c": None}), torch.tensor([0, 2, 1])),
        (Question(type="score", instructions="i", criteria=["l0", "l1", "l2", "l3"]), torch.tensor([3, 0, 2])),
        (Question(type="noul", instructions="i"), torch.tensor([1, 0, 1])),
    ]:
        m = DecisionModel(q, spec, bb, max_len=32)
        logits = m(ids, mask)
        assert logits.shape == (3, q.num_outputs)
        loss = m.loss(logits, y, ordinal_sigma=0.5)
        assert torch.isfinite(loss)
        p = m.probs(logits)
        ans = m.answer(p[0])
        if q.type == "choice":
            assert ans.choice in q.options and abs(sum(ans.probabilities.values()) - 1) < 1e-4
        elif q.type == "score":
            assert 0 <= ans.score <= 3 and ans.legend["3"] == "l3"
        else:
            assert 0 <= ans.noul <= 1
