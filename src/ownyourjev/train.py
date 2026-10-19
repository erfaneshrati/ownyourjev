"""Training loop for DecisionModel: frozen-embedding head, LoRA, or full fine-tuning.

Model selection (early stopping, temperature) uses only the validation examples passed in, which in the
benchmark are held-out *silver* labels, never gold.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ownyourjev.calibrate import fit_temperature
from ownyourjev.data import Example
from ownyourjev.evaluate import evaluate, primary_metric_name
from ownyourjev.models.backbones import resolve_backbone
from ownyourjev.models.decision import DecisionModel, load_backbone
from ownyourjev.spec import Question, label_to_index, render_state

Method = Literal["frozen", "lora", "full"]


@dataclass
class TrainConfig:
    backbone: str = "ultrabert"
    method: Method = "full"
    lr: float | None = None
    epochs: int = 5
    batch_size: int = 16
    max_len: int = 512
    warmup_frac: float = 0.06
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    ordinal_sigma: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dropout: float = 0.1
    pooling: str | None = None
    seed: int = 0
    patience: int = 2
    bf16: bool = True
    grad_accum: int = 1
    device: str = "auto"
    eval_batch_size: int = 64
    frozen_C: float | None = None  # None -> small grid chosen on val
    calibrate: bool = True
    max_steps: int | None = None  # for HPO pruning / smoke tests
    soft_labels: bool = False  # train on meta["soft"][qid] distributions when present (distillation)
    gpu_need_gb: float = 4.0  # free memory to wait for when device="auto" (cross-process GPU slots)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    model: DecisionModel
    tokenizer: Any
    config: TrainConfig
    n_train: int
    n_val: int
    val_loss: float
    val_metric: float
    val_metrics: dict
    history: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    train_s: float = 0.0
    best_epoch: int = 0
    temperature: float = 1.0

    def summary(self) -> dict:
        return {
            "backbone": self.config.backbone,
            "method": self.config.method,
            "lr": self.config.lr,
            "epochs": self.config.epochs,
            "n_train": self.n_train,
            "val_loss": self.val_loss,
            "val_metric": self.val_metric,
            "best_epoch": self.best_epoch,
            "wall_s": self.wall_s,
            "train_s": self.train_s,
            "temperature": self.temperature,
        }


# ---- data -------------------------------------------------------------------------------------------


def _targets(question: Question, examples: list[Example], qid: str) -> np.ndarray:
    return np.array([label_to_index(question, ex.labels[qid]) for ex in examples])  # type: ignore[index]


def _soft_targets(question: Question, examples: list[Example], qid: str) -> list:
    """Soft target per example (distribution over options, or P(true) for noul); one-hot when missing."""
    out = []
    for ex in examples:
        soft = (ex.meta.get("soft") or {}).get(qid)
        if soft is None:
            i = label_to_index(question, ex.labels[qid])  # type: ignore[index]
            soft = float(i) if question.type == "noul" else [1.0 if j == i else 0.0 for j in range(question.num_outputs)]
        out.append(soft)
    return out


def _texts(examples: list[Example]) -> list[str]:
    return [render_state(ex.state) for ex in examples]


class _Collate:
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[tuple[str, object]]):
        texts, ys = zip(*batch)
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        y = torch.as_tensor(ys, dtype=torch.float32) if isinstance(ys[0], (list, float)) else torch.as_tensor(ys)
        return enc["input_ids"], enc["attention_mask"], y


def _autocast(cfg: TrainConfig):
    if cfg.device.startswith("cuda") and cfg.bf16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def model_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)


@torch.no_grad()
def predict_logits(
    model: DecisionModel, tokenizer, texts: list[str], *, max_len: int, batch_size: int = 64, device: str | None = None, bf16: bool = True
) -> np.ndarray:
    model.eval()
    device = device or model_device(model)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    cfg = TrainConfig(device=device, bf16=bf16)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with _autocast(cfg):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for i, row in zip(idx, logits.float().cpu().numpy()):
            out[i] = row
    return np.stack(out)


def predict_probs(model: DecisionModel, tokenizer, examples: list[Example], *, batch_size: int = 64, device: str | None = None) -> np.ndarray:
    logits = predict_logits(model, tokenizer, _texts(examples), max_len=model.max_len, batch_size=batch_size, device=device or model_device(model))
    return model.probs(torch.as_tensor(logits)).numpy()


EMBED_CACHE: dict[tuple, np.ndarray] = {}  # (hf_id, pooling, max_len, text) -> vector; frozen-head ladders reuse it


@torch.no_grad()
def embed_texts(model: DecisionModel, tokenizer, texts: list[str], *, max_len: int, batch_size: int = 64, device: str | None = None, use_cache: bool = True) -> np.ndarray:
    model.eval()
    device = device or model_device(model)
    key = (model.spec.hf_id, model.pooling, max_len)
    out: list[np.ndarray | None] = [EMBED_CACHE.get(key + (t,)) if use_cache else None for t in texts]
    todo = [i for i, v in enumerate(out) if v is None]
    order = sorted(todo, key=lambda i: len(texts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            emb = model.embed(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for i, row in zip(idx, emb.float().cpu().numpy()):
            out[i] = row
            if use_cache:
                EMBED_CACHE[key + (texts[i],)] = row
    return np.stack(out)  # type: ignore[arg-type]


# ---- training ---------------------------------------------------------------------------------------


def _most_free_gpu() -> tuple[int | None, float]:
    best, best_free = None, 0.0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best, best_free = i, free
    return best, best_free / 1e9


def pick_device(min_free_gb: float = 2.0, wait_s: float = 600.0) -> str:
    """The CUDA device with the most free memory. Never silently falls back to CPU when a GPU exists:
    if every GPU has less than ``min_free_gb`` free (another job is using it), wait up to ``wait_s`` for
    memory to free up, then take the best one anyway (an OOM is caught and retried by the callers)."""
    if not torch.cuda.is_available():
        return "cpu"
    t0 = time.time()
    while True:
        best, free_gb = _most_free_gpu()
        if best is None:
            return "cpu"
        if free_gb >= min_free_gb or time.time() - t0 > wait_s:
            return f"cuda:{best}"
        time.sleep(10)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _val_eval(model: DecisionModel, tokenizer, question: Question, val: list[Example], qid: str, cfg: TrainConfig):
    logits = predict_logits(model, tokenizer, _texts(val), max_len=cfg.max_len, batch_size=cfg.eval_batch_size, device=cfg.device, bf16=cfg.bf16)
    y = _targets(question, val, qid)
    with torch.no_grad():
        loss = float(model.loss(torch.as_tensor(logits), torch.as_tensor(y), label_smoothing=0.0, ordinal_sigma=0.0))
    probs = model.probs(torch.as_tensor(logits), calibrated=False).numpy()
    golds = [ex.labels[qid] for ex in val]  # type: ignore[index]
    metrics = evaluate(question, golds, probs)
    return loss, metrics, logits, y


def _build_model(question: Question, cfg: TrainConfig) -> tuple[DecisionModel, Any]:
    spec = resolve_backbone(cfg.backbone)
    backbone, tok = load_backbone(spec)
    if cfg.method == "lora":
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=spec.lora_targets, bias="none")
        backbone = get_peft_model(backbone, lcfg)
    model = DecisionModel(question, spec, backbone, pooling=cfg.pooling, dropout=cfg.dropout, max_len=min(cfg.max_len, spec.max_len))
    return model.to(cfg.device), tok


def train(question: Question, train_ex: list[Example], val_ex: list[Example], cfg: TrainConfig, *, qid: str = "answer", verbose: bool = False) -> TrainResult:
    t_wall = time.time()
    if cfg.device == "auto" and os.environ.get("OYJ_GPU_SCHEDULER", "1") == "1" and torch.cuda.is_available():
        from ownyourjev.gpu import gpu_slot

        with gpu_slot(cfg.gpu_need_gb) as dev:
            return train(question, train_ex, val_ex, TrainConfig(**{**cfg.to_dict(), "device": dev}), qid=qid, verbose=verbose)
    if cfg.device == "auto":
        cfg = TrainConfig(**{**cfg.to_dict(), "device": pick_device()})
    if cfg.device.startswith("cuda"):
        torch.cuda.set_device(cfg.device)  # keep optimizer/stream calls off a foreign, possibly full, GPU
    _seed(cfg.seed)
    train_ex = [ex for ex in train_ex if ex.labels is not None and qid in ex.labels]
    val_ex = [ex for ex in val_ex if ex.labels is not None and qid in ex.labels]
    model, tok = _build_model(question, cfg)
    if cfg.method == "frozen":
        res = _train_frozen(model, tok, question, train_ex, val_ex, cfg, qid)
    else:
        res = _train_gradient(model, tok, question, train_ex, val_ex, cfg, qid, verbose=verbose)
    res.wall_s = time.time() - t_wall
    return res


def train_with_oom_retry(question: Question, train_ex: list[Example], val_ex: list[Example], cfg: TrainConfig, *, qid: str = "answer", log=print) -> TrainResult | None:
    """``train`` with a retry on CUDA out-of-memory: free the cache, halve the batch, and let ``device="auto"``
    re-pick the GPU with the most free memory (another process may have grabbed the first one)."""
    import gc

    for attempt in range(4):
        try:
            return train(question, train_ex, val_ex, cfg, qid=qid)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 3:
                return None
            bs = max(1, cfg.batch_size // 2)
            log(f"    out of memory on {cfg.device}: retrying with batch_size={bs}, grad_accum={cfg.grad_accum * 2}")
            cfg = TrainConfig(**{**cfg.to_dict(), "batch_size": bs, "grad_accum": cfg.grad_accum * 2, "eval_batch_size": max(1, cfg.eval_batch_size // 2), "device": "auto"})
    return None


def _train_frozen(model, tok, question, train_ex, val_ex, cfg, qid) -> TrainResult:
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    x_tr = embed_texts(model, tok, _texts(train_ex), max_len=cfg.max_len, batch_size=cfg.eval_batch_size, device=cfg.device)
    x_va = embed_texts(model, tok, _texts(val_ex), max_len=cfg.max_len, batch_size=cfg.eval_batch_size, device=cfg.device)
    y_tr = _targets(question, train_ex, qid)
    y_va = _targets(question, val_ex, qid)
    k = question.num_outputs if question.type != "noul" else 2
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xs_tr, xs_va = (x_tr - mu) / sd, (x_va - mu) / sd
    grid = [cfg.frozen_C] if cfg.frozen_C else [0.01, 0.1, 1.0, 10.0]
    best = None
    history = []
    for C in grid:
        clf = LogisticRegression(C=C, max_iter=3000)
        if len(np.unique(y_tr)) < 2:
            break
        clf.fit(xs_tr, y_tr)
        w_full = np.zeros((k, x_tr.shape[1]), dtype=np.float32)
        b_full = np.full((k,), -20.0, dtype=np.float32)
        if len(clf.classes_) == 2 and question.type == "noul":
            w_full = clf.coef_.astype(np.float32)
            b_full = clf.intercept_.astype(np.float32)
            k_eff = 1
        elif len(clf.classes_) == 2:
            # sklearn binary: one row = logit(class1) - logit(class0)
            w_full[clf.classes_[1]] = clf.coef_[0]
            b_full[clf.classes_[1]] = clf.intercept_[0]
            w_full[clf.classes_[0]] = 0.0
            b_full[clf.classes_[0]] = 0.0
            k_eff = k
        else:
            for row, c in enumerate(clf.classes_):
                w_full[c] = clf.coef_[row]
                b_full[c] = clf.intercept_[row]
            k_eff = k
        # fold standardisation into the linear layer: w' = w/sd, b' = b - w.mu/sd
        w_lin = (w_full / sd[None, :]).astype(np.float32)
        b_lin = (b_full - (w_full * (mu / sd)[None, :]).sum(1)).astype(np.float32)
        head = torch.nn.Linear(x_tr.shape[1], k_eff)
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(w_lin[:k_eff]))
            head.bias.copy_(torch.as_tensor(b_lin[:k_eff]))
        model.head = head.to(cfg.device)
        logits = (xs_va @ w_full.T + b_full)[:, :k_eff]
        with torch.no_grad():
            loss = float(model.loss(torch.as_tensor(logits), torch.as_tensor(y_va)))
        probs = model.probs(torch.as_tensor(logits), calibrated=False).numpy()
        metrics = evaluate(question, [ex.labels[qid] for ex in val_ex], probs)
        history.append({"C": C, "val_loss": loss, **{f"val_{k2}": v for k2, v in metrics.items()}})
        if best is None or loss < best[0]:
            best = (loss, metrics, copy.deepcopy(head.state_dict()), logits, C)
    if best is None:  # degenerate (single class): keep zero head
        logits = np.zeros((len(val_ex), model.head.out_features))
        best = (math.nan, evaluate(question, [ex.labels[qid] for ex in val_ex], model.probs(torch.as_tensor(logits)).numpy()), copy.deepcopy(model.head.state_dict()), logits, None)
    model.head.load_state_dict(best[2])
    temp = 1.0
    if cfg.calibrate and len(val_ex) >= 8:
        temp = fit_temperature(question, best[3], y_va)
        model.temperature.fill_(temp)
    return TrainResult(model, tok, cfg, len(train_ex), len(val_ex), best[0], best[1].get(primary_metric_name(question), math.nan), best[1], history, 0.0, time.time() - t0, 0, temp)


def _train_gradient(model, tok, question, train_ex, val_ex, cfg, qid, *, verbose: bool) -> TrainResult:
    t0 = time.time()
    spec = model.spec
    lr = cfg.lr or (spec.lora_lr if cfg.method == "lora" else spec.default_lr)
    y_tr = _targets(question, train_ex, qid)
    ys = _soft_targets(question, train_ex, qid) if cfg.soft_labels else y_tr.tolist()
    ds = list(zip(_texts(train_ex), ys))
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_Collate(tok, cfg.max_len), generator=g, drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    head_params = set(id(p) for p in model.head.parameters())
    groups = [
        {"params": [p for p in decay if id(p) not in head_params], "weight_decay": cfg.weight_decay, "lr": lr},
        {"params": [p for p in no_decay if id(p) not in head_params], "weight_decay": 0.0, "lr": lr},
        {"params": list(model.head.parameters()), "weight_decay": 0.0, "lr": max(lr, 1e-4) if cfg.method == "lora" else lr * 5},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)
    best_loss, best_state, best_epoch, best_metrics, best_logits = math.inf, None, 0, {}, None
    history: list[dict] = []
    step = 0
    bad = 0
    y_va = _targets(question, val_ex, qid)
    for epoch in range(cfg.epochs):
        model.train()
        run_loss, n_b = 0.0, 0
        for i, (ids, mask, ys) in enumerate(dl):
            with _autocast(cfg):
                logits = model(ids.to(cfg.device), mask.to(cfg.device))
            loss = model.loss(logits, ys.to(cfg.device), label_smoothing=cfg.label_smoothing, ordinal_sigma=cfg.ordinal_sigma) / cfg.grad_accum
            loss.backward()
            run_loss += float(loss) * cfg.grad_accum
            n_b += 1
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if cfg.max_steps and step >= cfg.max_steps:
                    break
        val_loss, metrics, logits_va, _ = _val_eval(model, tok, question, val_ex, qid, cfg)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(1, n_b), "val_loss": val_loss, **{f"val_{k}": v for k, v in metrics.items()}, "elapsed_s": time.time() - t0}
        history.append(rec)
        if verbose:
            print(rec)
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, best_metrics, best_logits, bad = val_loss, epoch + 1, metrics, logits_va, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if cfg.max_steps and step >= cfg.max_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    temp = 1.0
    if cfg.calibrate and best_logits is not None and len(val_ex) >= 8:
        temp = fit_temperature(question, best_logits, y_va)
        model.temperature.fill_(temp)
    model.eval()
    return TrainResult(model, tok, cfg, len(train_ex), len(val_ex), best_loss, best_metrics.get(primary_metric_name(question), math.nan), best_metrics, history, 0.0, time.time() - t0, best_epoch, temp)
