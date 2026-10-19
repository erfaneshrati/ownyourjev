import numpy as np
import pytest

from ownyourjev.calibration import Calibrator, calibration_metrics, cv_scores, ece, select_calibrator


def _overconfident_binary(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-3 * z))  # 3x overconfident
    return p, y


def _overconfident_multi(n=2000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 1.5, (n, k))
    P_true = np.exp(L) / np.exp(L).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in P_true])
    P = np.exp(3 * L) / np.exp(3 * L).sum(1, keepdims=True)
    return P, y


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_binary_calibrators_reduce_ece(method):
    p, y = _overconfident_binary()
    c = Calibrator(method).fit(p[:1000], y[:1000])
    assert ece(c.apply(p[1000:]), y[1000:]) < ece(p[1000:], y[1000:]) * 0.6


@pytest.mark.parametrize("method", ["temperature", "vector", "isotonic"])
def test_multiclass_calibrators_reduce_ece(method):
    P, y = _overconfident_multi()
    c = Calibrator(method).fit(P[:1000], y[:1000])
    Q = c.apply(P[1000:])
    assert np.allclose(Q.sum(1), 1)
    assert ece(Q, y[1000:]) < ece(P[1000:], y[1000:]) * 0.6


def test_temperature_keeps_decisions_and_roundtrips():
    P, y = _overconfident_multi()
    c = Calibrator("temperature").fit(P, y)
    assert c.params["T"] > 1.5
    assert (c.apply(P).argmax(1) == P.argmax(1)).all()
    c2 = Calibrator.from_dict(c.to_dict())
    assert np.allclose(c2.apply(P), c.apply(P))
    p, yb = _overconfident_binary()
    cb = Calibrator("temperature").fit(p, yb)
    assert ((cb.apply(p) >= 0.5) == (p >= 0.5)).all()


def test_selection_prefers_calibration_when_needed():
    p, y = _overconfident_binary(400)
    c, scores = select_calibrator(p, y)
    assert c.method != "none" and scores[c.method]["nll"] < scores["none"]["nll"]
    m = calibration_metrics(p, y)
    assert {"ece", "ace", "brier", "nll", "reliability"} <= set(m)


def test_cv_scores_with_groups():
    P, y = _overconfident_multi(300, 3)
    groups = np.arange(300) // 10
    s = cv_scores(P, y, groups=groups)
    assert "temperature" in s


def test_fit_for_question_and_answer_respect_argmax():
    import torch

    from ownyourjev.calibration import fit_for_question
    from ownyourjev.models.decision import DecisionModel
    from ownyourjev.spec import Question

    rng = np.random.default_rng(0)
    q = Question(type="choice", instructions="pick", criteria={"a": "A", "b": "B", "c": "C"})
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3)) + 3.0 * np.eye(3)[y]  # sharp, overconfident model
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    cal, scores = fit_for_question(q, p, [q.options[i] for i in y])
    assert set(scores) <= {"none", "temperature"}
    m = DecisionModel.__new__(DecisionModel)
    m.question, m.calibrator = q, cal
    for row in p[:20]:
        ans = m.answer(torch.as_tensor(row))
        assert ans.choice == q.options[int(row.argmax())]
        assert abs(sum(ans.probabilities.values()) - 1) < 1e-4

    qn = Question(type="noul", instructions="true?")
    yb = rng.integers(0, 2, 400)
    pb = 1 / (1 + np.exp(-(4.0 * (2 * yb - 1) + rng.normal(scale=2, size=400))))
    calb, _ = fit_for_question(qn, pb, [bool(v) for v in yb])
    m.question, m.calibrator = qn, calb
    assert 0.0 <= m.answer(float(pb[0])).noul <= 1.0
