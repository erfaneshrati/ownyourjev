"""Shared-encoder, two-head (toxic, jailbreak) noul model on gte-modernbert, with soft-label BCE,
ONNX export (one session answers both heads) and prediction helpers."""
import json, math, time, random, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from common import Q, HEADS
from ownyourjev.spec import render_state

HF = "Alibaba-NLP/gte-modernbert-base"


class MH(nn.Module):
    def __init__(self, backbone, n=2, dropout=0.1, max_len=512):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.config.hidden_size, n)
        self.max_len = max_len
        self.register_buffer("temperature", torch.ones(n))

    def forward(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(self.dropout(h))

    def save(self, path, tok):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "backbone"); tok.save_pretrained(path / "backbone")
        torch.save({"head": self.head.state_dict(), "temperature": self.temperature.cpu()}, path / "head.pt")
        meta = {"question": Q["toxic"].model_dump(), "heads": HEADS, "backbone": "gte-modernbert", "hf_id": HF,
                "pooling": "cls", "max_len": self.max_len, "multihead": True,
                "note": "probs output is (batch, 2): [:,0]=P(toxic), [:,1]=P(jailbreak)"}
        (path / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path, device="cpu", attn=None):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        kw = {"attn_implementation": attn} if attn else {}
        bb = AutoModel.from_pretrained(path / "backbone", dtype=torch.float32, **kw)
        m = cls(bb, max_len=meta["max_len"])
        st = torch.load(path / "head.pt", map_location="cpu")
        m.head.load_state_dict(st["head"]); m.temperature.copy_(st["temperature"])
        return m.to(device).eval(), AutoTokenizer.from_pretrained(path / "backbone")


@torch.no_grad()
def predict_logits(m, tok, texts, device, bs=64):
    m.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=m.max_len, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            lg = m(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for i, r in zip(idx, lg.float().cpu().numpy()):
            out[i] = r
    return np.stack(out)


def texts_of(exs):
    return [render_state(e.state) for e in exs]


def train_mh(train, val, *, epochs=4, lr=3e-5, bs=16, max_len=512, seed=0, pos_weight=(1.0, 1.0), device="cuda:1",
             log=print, warmup=0.06, wd=0.01, accum=2):
    """train/val: lists of (text, [y_toxic, y_jb]) with soft targets in [0,1]. Early stopping on val BCE (silver)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(HF)
    bb = AutoModel.from_pretrained(HF, dtype=torch.float32)
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = MH(bb, max_len=max_len).to(device)
    pw = torch.tensor(pos_weight, device=device)
    head_ids = {id(p) for p in m.head.parameters()}
    groups = [
        {"params": [p for n, p in m.named_parameters() if id(p) not in head_ids and p.ndim > 1], "weight_decay": wd, "lr": lr},
        {"params": [p for n, p in m.named_parameters() if id(p) not in head_ids and p.ndim <= 1], "weight_decay": 0.0, "lr": lr},
        {"params": list(m.head.parameters()), "weight_decay": 0.0, "lr": lr * 5},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)
    steps = math.ceil(len(train) / (bs * accum)) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    vt = [t for t, _ in val]; vy = torch.tensor([y for _, y in val], dtype=torch.float32)
    best = (math.inf, None, 0)
    g = random.Random(seed)
    for ep in range(epochs):
        m.train(); idx = list(range(len(train))); g.shuffle(idx)
        tl, nb = 0.0, 0
        for k, s in enumerate(range(0, len(idx), bs)):
            b = [train[i] for i in idx[s:s + bs]]
            enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            y = torch.tensor([yy for _, yy in b], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = F.binary_cross_entropy_with_logits(lg.float(), y, pos_weight=pw)
            (loss / accum).backward()
            if (k + 1) % accum == 0 or s + bs >= len(idx):
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tl += float(loss); nb += 1
        lv = torch.as_tensor(predict_logits(m, tok, vt, device))
        vl = float(F.binary_cross_entropy_with_logits(lv, vy))
        log(f"  epoch {ep+1} train {tl/nb:.4f} val_bce {vl:.4f}")
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep + 1)
    m.load_state_dict(best[1])
    m.eval()
    return m, tok, best[2]


class _W(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, ids, mask):
        return torch.sigmoid(self.m(ids, mask) / self.m.temperature)


def export_onnx(model_dir, int8=True, attn=None, out_name="model.onnx"):
    from ownyourjev.export import quantize_int8
    m, tok = MH.load(model_dir, attn=attn)
    enc = tok(["example input for tracing", "second"], padding=True, return_tensors="pt")
    out = Path(model_dir) / out_name
    torch.onnx.export(_W(m), (enc["input_ids"], enc["attention_mask"]), str(out), input_names=["input_ids", "attention_mask"],
                      output_names=["probs"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    if int8:
        quantize_int8(out)
    return out


def onnx_probs(model_dir, texts, onnx_name="model.int8.onnx", threads=8):
    import onnxruntime as ort
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "backbone")
    o = ort.SessionOptions(); o.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(Path(model_dir) / onnx_name), o, providers=["CPUExecutionProvider"])
    out = []
    for t in texts:
        enc = tok([t], truncation=True, max_length=meta["max_len"], return_tensors="np")
        out.append(sess.run(["probs"], {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0])
    return np.stack(out)
