"""Shared-encoder multi-head scorer for SummEval: one (optionally layer-truncated) gte-modernbert pass over
"summary + truncated article", CLS embedding + cheap lexical alignment features over the FULL article -> 4 heads x 5 levels.
Library extension candidate: ownyourjev.models.multihead (several Score heads on one encoder + side features)."""
from __future__ import annotations
import json, math, os, random, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import DIMS, text_of, render
from feats import features, FEAT_NAMES

HF = "Alibaba-NLP/gte-modernbert-base"
K = 5


@dataclass
class Cfg:
    n_layers: int = 22
    max_len: int = 256
    use_feats: bool = True
    lr: float = 3e-5
    head_lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 16
    warmup: float = 0.06
    wd: float = 0.01
    rank_w: float = 0.0      # weight of within-article pairwise ranking loss
    seed: int = 0
    gold_w: float = 1.0      # loss weight of human-labelled (dev/seed) examples
    syn_w: float = 1.0       # loss weight of synthetic examples
    syn_dims: str = ""       # comma list of dims synthetic examples train (empty = all)
    syn_drop: str = ""       # comma list of synthetic profiles to drop
    dropout: float = 0.1
    pool: str = "cls"        # cls | clsmean (concat CLS and masked mean)
    aux_w: float = 0.0       # weight of the auxiliary sentence-verification head (target meta["verify"]["v_mean"])
    aux_mix: float = 0.0     # inference: consistency expected score += aux_mix * 4 * sigmoid(aux)  (0 = aux is training-only)
    ctx_chars: int = 700     # article-context budget for ctx=align
    ctx: str = "lead"        # lead | align (article sentences chosen by overlap with the summary)
    dims: list = field(default_factory=lambda: list(DIMS))


class MH(nn.Module):
    def __init__(self, cfg: Cfg, backbone=None):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            c = AutoConfig.from_pretrained(HF, num_hidden_layers=cfg.n_layers)
            c.layer_types = c.layer_types[: cfg.n_layers]
            backbone = AutoModel.from_pretrained(HF, config=c, dtype=torch.float32)
        self.backbone = backbone
        h = backbone.config.hidden_size
        nf = len(FEAT_NAMES) if cfg.use_feats else 0
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h * (2 if cfg.pool == "clsmean" else 1), len(cfg.dims) * K)
        self.fhead = nn.Sequential(nn.Linear(nf, 64), nn.GELU(), nn.Linear(64, len(cfg.dims) * K)) if nf else None
        self.aux = nn.Linear(h * (2 if cfg.pool == "clsmean" else 1), 1) if cfg.aux_w > 0 else None
        self.aux_last = None
        self.register_buffer("f_mu", torch.zeros(max(nf, 1)))
        self.register_buffer("f_sd", torch.ones(max(nf, 1)))
        self.register_buffer("temp", torch.ones(len(cfg.dims)))

    def forward(self, ids, mask, feats=None):
        hs = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        hcls = self.pool_h(hs, mask)
        z = self.head(self.drop(hcls))
        if self.aux is not None:
            self.aux_last = self.aux(self.drop(hcls)).squeeze(-1)
        if self.fhead is not None:
            z = z + self.fhead((feats - self.f_mu) / self.f_sd)
        return z.view(-1, len(self.cfg.dims), K)

    def pool_h(self, hs, mask):
        if self.cfg.pool == "clsmean":
            m = mask.unsqueeze(-1).to(hs.dtype)
            return torch.cat([hs[:, 0], (hs * m).sum(1) / m.sum(1).clamp(min=1.0)], -1)
        return hs[:, 0]

    def probs(self, z):
        return torch.softmax(z.float() / self.temp.view(1, -1, 1), -1)


def soft_from_example(e, dims):
    """(D, K) soft target: LLM distribution in meta['soft'] or a gold float -> linear interpolation."""
    out = []
    for d in dims:
        s = (e.meta.get("soft") or {}).get(d)
        if s is None:
            v = float(e.labels[d]); lo = int(math.floor(v)); hi = min(lo + 1, K - 1); w = v - lo
            s = [0.0] * K; s[lo] += 1 - w; s[hi] += w
        out.append(s)
    return out


def featurize(exs):
    return np.stack([features(e.state["summary"], e.state["source_document"]) for e in exs]).astype(np.float32)


class Batcher:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def __call__(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, tok, exs, device, bs=64, feats=None):
    model.eval()
    texts = [render(e.state, model.cfg.ctx, model.cfg.ctx_chars) for e in exs]
    feats = featurize(exs) if feats is None else feats
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(exs), len(model.cfg.dims), K), dtype=np.float32)
    aux_out = np.full(len(exs), np.nan, dtype=np.float32)
    b = Batcher(tok, model.cfg.max_len)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        ids, m = b([texts[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            z = model(ids.to(device), m.to(device), torch.as_tensor(feats[idx]).to(device))
        out[idx] = model.probs(z).cpu().numpy()
        if model.aux is not None:
            aux_out[idx] = torch.sigmoid(model.aux_last.float()).cpu().numpy()
    model.last_aux = aux_out
    return out


def train(cfg: Cfg, train_exs, device, *, log=print, eval_fn=None):
    """train_exs: Examples with meta['soft'] (LLM) or float gold labels; group used for the ranking loss."""
    _seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(HF)
    model = MH(cfg).to(device)
    drop = set(filter(None, cfg.syn_drop.split(",")))
    train_exs = [e for e in train_exs if e.meta.get("profile") not in drop]
    X = featurize(train_exs)
    if cfg.use_feats:
        model.f_mu.copy_(torch.as_tensor(X.mean(0))); model.f_sd.copy_(torch.as_tensor(X.std(0) + 1e-4))
    Y = np.array([soft_from_example(e, cfg.dims) for e in train_exs], dtype=np.float32)
    sd = set(filter(None, cfg.syn_dims.split(","))) or set(cfg.dims)
    Wt = np.array([[(cfg.syn_w if d in sd else 0.0) if e.meta.get("synthetic") else (cfg.gold_w if "soft" not in e.meta else 1.0) for d in cfg.dims] for e in train_exs], dtype=np.float32)
    texts = [render(e.state, cfg.ctx, cfg.ctx_chars) for e in train_exs]
    A = np.array([((e.meta.get("verify") or {}).get("v_mean", np.nan)) for e in train_exs], dtype=np.float32)
    groups = [e.group for e in train_exs]
    # batches: group-aware (several summaries of the same article per batch) so the ranking loss has pairs
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    bt = Batcher(tok, cfg.max_len)
    head_p = list(model.head.parameters()) + (list(model.fhead.parameters()) if model.fhead is not None else [])
    hid = {id(p) for p in head_p}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.ndim > 1], "lr": cfg.lr, "weight_decay": cfg.wd},
        {"params": [p for p in model.backbone.parameters() if p.ndim <= 1], "lr": cfg.lr, "weight_decay": 0.0},
        {"params": head_p, "lr": cfg.head_lr, "weight_decay": 0.0}], betas=(0.9, 0.98), eps=1e-6)
    rng = random.Random(cfg.seed)

    def make_batches():
        chunks = []
        for g, idx in by.items():
            idx = idx[:]; rng.shuffle(idx)
            for s in range(0, len(idx), 4):
                chunks.append(idx[s:s + 4])
        rng.shuffle(chunks)
        bs, cur = [], []
        for c in chunks:
            cur += c
            if len(cur) >= cfg.batch_size:
                bs.append(cur); cur = []
        if cur: bs.append(cur)
        return bs

    n_steps = len(make_batches()) * cfg.epochs
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup * n_steps), n_steps)
    lv = torch.arange(K, device=device, dtype=torch.float32)
    hist = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train(); tot = 0.0; nb = 0
        for b in make_batches():
            ids, m = bt([texts[i] for i in b])
            y = torch.as_tensor(Y[b]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                z = model(ids.to(device), m.to(device), torch.as_tensor(X[b]).to(device))
            logp = F.log_softmax(z.float(), -1)
            w = torch.as_tensor(Wt[b]).to(device)
            loss = (-(y * logp).sum(-1) * w).sum() / w.sum().clamp(min=1e-6)
            if model.aux is not None:
                at = torch.as_tensor(A[b]).to(device); ok = ~torch.isnan(at)
                if ok.any():
                    loss = loss + cfg.aux_w * F.binary_cross_entropy_with_logits(model.aux_last.float()[ok], at[ok])
            if cfg.rank_w > 0:
                s = (logp.exp() * lv).sum(-1)  # (B, D)
                t = (y * lv).sum(-1)
                g = [groups[i] for i in b]
                same = torch.tensor([[gi == gj for gj in g] for gi in g], device=device)
                dt = t[:, None, :] - t[None, :, :]
                ds = s[:, None, :] - s[None, :, :]
                msk = same[:, :, None] & (dt > 0.25)
                if msk.any():
                    rl = F.softplus(-(ds[msk]) * 2.0).mean()
                    loss = loss + cfg.rank_w * rl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += float(loss); nb += 1
        rec = {"epoch": ep + 1, "loss": tot / max(1, nb), "s": round(time.time() - t0)}
        if eval_fn is not None:
            rec.update(eval_fn(model, tok))
        hist.append(rec); log(rec)
    return model, tok, hist


def save(model, tok, path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    model.backbone.config.save_pretrained(path / "backbone")
    torch.save(model.state_dict(), path / "mh.pt")
    tok.save_pretrained(path / "backbone")
    (path / "mh_cfg.json").write_text(json.dumps(asdict(model.cfg)))


def load(path, device="cpu", attn="eager"):
    path = Path(path)
    cfg = Cfg(**json.loads((path / "mh_cfg.json").read_text()))
    c = AutoConfig.from_pretrained(path / "backbone")
    c._attn_implementation = attn
    bb = AutoModel.from_config(c, attn_implementation=attn)
    m = MH(cfg, backbone=bb)
    m.load_state_dict(torch.load(path / "mh.pt", map_location="cpu"))
    return m.to(device).eval(), AutoTokenizer.from_pretrained(path / "backbone")
