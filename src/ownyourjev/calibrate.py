"""Temperature scaling fitted on held-out (silver) logits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ownyourjev.spec import Question


def fit_temperature(question: Question, logits: np.ndarray, targets: np.ndarray, *, max_iter: int = 200) -> float:
    """Return the temperature T > 0 minimising NLL of logits / T on the given targets."""
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets))
    # a perfectly separated (small) validation set drives T -> 0 (ever sharper); keep T = 1 there
    pred = (z.reshape(-1) > 0).long() if question.type == "noul" else z.argmax(-1)
    if len(y) < 30 or bool((pred == y.long()).all()):
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def nll() -> torch.Tensor:
        t = torch.exp(log_t)
        if question.type == "noul":
            return F.binary_cross_entropy_with_logits(z.reshape(-1) / t, y.float())
        return F.cross_entropy(z / t, y.long())

    def closure():
        opt.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0
    t = float(torch.exp(log_t).item())
    if not np.isfinite(t) or t <= 0:
        return 1.0
    return float(np.clip(t, 0.5, 10.0))
