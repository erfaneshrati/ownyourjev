"""Lean ONNX export of a gte-modernbert DecisionModel for CPU inference.

The stock HF export of ModernBERT spends >50% of CPU time in mask plumbing (Where/IsNaN/Slice per layer).
This wrapper re-implements the ModernBERT forward (same weights, same math) with additive masks computed once,
eager attention and precomputed RoPE tables, so the graph is just MatMuls + elementwise ops.
Output = DecisionModel.probs (sigmoid/softmax with temperature), inputs input_ids / attention_mask, so it is a
drop-in `model.onnx` for ownyourjev.export.OnnxPredictor and benchmarks.v2.latency.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ownyourjev.models.decision import DecisionModel


class LeanModernBert(nn.Module):
    def __init__(self, m: DecisionModel, max_pos: int = 1024):
        super().__init__()
        self.m = m
        bb = m.backbone
        cfg = bb.config
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // self.nh
        self.win = cfg.local_attention // 2
        self.types = list(cfg.layer_types)
        pos = torch.arange(max_pos).float()
        for t in set(self.types):
            base = cfg.rope_parameters[t]["rope_theta"]
            inv = 1.0 / (base ** (torch.arange(0, self.hd, 2).float() / self.hd))
            fr = pos[:, None] * inv[None, :]
            emb = torch.cat([fr, fr], -1)
            self.register_buffer(f"cos_{t}", emb.cos(), persistent=False)
            self.register_buffer(f"sin_{t}", emb.sin(), persistent=False)
        h2 = self.hd // 2
        P = torch.zeros(self.hd, self.hd)  # x @ P == rotate_half(x)
        P[torch.arange(h2) + h2, torch.arange(h2)] = -1.0
        P[torch.arange(h2), torch.arange(h2) + h2] = 1.0
        self.register_buffer("P", P, persistent=False)
        # split fused weights so the graph needs no Slice/Split copies
        D = cfg.hidden_size
        self.wq, self.wk, self.wv, self.wi_in, self.wi_gate = (nn.ModuleList() for _ in range(5))
        for layer in bb.layers:
            W = layer.attn.Wqkv.weight.data
            b = layer.attn.Wqkv.bias
            for i, ml in enumerate([self.wq, self.wk, self.wv]):
                lin = nn.Linear(D, D, bias=b is not None)
                lin.weight.data = W[i * D : (i + 1) * D].clone()
                if b is not None:
                    lin.bias.data = b.data[i * D : (i + 1) * D].clone()
                ml.append(lin)
            Wi = layer.mlp.Wi.weight.data
            bi = layer.mlp.Wi.bias
            I = Wi.shape[0] // 2
            for i, ml in enumerate([self.wi_in, self.wi_gate]):
                lin = nn.Linear(D, I, bias=bi is not None)
                lin.weight.data = Wi[i * I : (i + 1) * I].clone()
                if bi is not None:
                    lin.bias.data = bi.data[i * I : (i + 1) * I].clone()
                ml.append(lin)

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def forward(self, input_ids, attention_mask):
        bb = self.m.backbone
        B, L = input_ids.shape
        h = bb.embeddings(input_ids=input_ids)
        am = attention_mask.to(h.dtype)
        full = (1.0 - am)[:, None, None, :] * -1e4  # B,1,1,L
        idx = torch.arange(L, device=h.device)
        far = ((idx[None, :] - idx[:, None]).abs() > self.win).to(h.dtype) * -1e4  # L,L
        masks = {"full_attention": full, "sliding_attention": full + far[None, None]}
        rope = {t: (getattr(self, f"cos_{t}")[:L], getattr(self, f"sin_{t}")[:L]) for t in set(self.types)}
        for li, (layer, t) in enumerate(zip(bb.layers, self.types)):
            x = layer.attn_norm(h)
            cos, sin = rope[t]
            q = self.wq[li](x).view(B, L, self.nh, self.hd).transpose(1, 2)
            k = self.wk[li](x).view(B, L, self.nh, self.hd).transpose(1, 2)
            v = self.wv[li](x).view(B, L, self.nh, self.hd).transpose(1, 2)
            q = q * cos + torch.matmul(q, self.P) * sin
            k = k * cos + torch.matmul(k, self.P) * sin
            a = torch.matmul(q, k.transpose(2, 3)) * (self.hd ** -0.5) + masks[t]
            a = torch.softmax(a, -1)
            o = torch.matmul(a, v).transpose(1, 2).reshape(B, L, -1)
            h = h + layer.attn.Wo(o)
            y = layer.mlp_norm(h)
            h = h + layer.mlp.Wo(layer.mlp.act(self.wi_in[li](y)) * self.wi_gate[li](y))
        h = bb.final_norm(h)
        if self.m.pooling == "cls":
            pooled = h[:, 0]
        else:
            mm = am.unsqueeze(-1)
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1.0)
        return self.m.probs(self.m.head(pooled))


ROPE_MODE = "matmul"


class LeanUltraBert(nn.Module):
    """Same idea for UltraBERT (Llama-style: RMSNorm, q/k/v/o, SiLU-GLU MLP, per-layer RoPE theta and window)."""

    def __init__(self, m: DecisionModel, max_pos: int = 1024):
        super().__init__()
        self.m = m
        cfg = m.backbone.config
        self.nh = cfg.num_attention_heads
        self.hd = cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.windows = [cfg.sliding_window if t == "sliding_attention" else None for t in cfg.layer_types]
        self.thetas = [cfg.rope_parameters[t]["rope_theta"] for t in cfg.layer_types]
        pos = torch.arange(max_pos).float()
        self.rope_names = {}
        for th in set(self.thetas):
            inv = 1.0 / (th ** (torch.arange(0, self.hd, 2).float() / self.hd))
            fr = pos[:, None] * inv[None, :]
            emb = torch.cat([fr, fr], -1)
            nm = str(int(th))
            self.rope_names[th] = nm
            self.register_buffer(f"cos_{nm}", emb.cos(), persistent=False)
            self.register_buffer(f"sin_{nm}", emb.sin(), persistent=False)
        h2 = self.hd // 2
        P = torch.zeros(self.hd, self.hd)
        P[torch.arange(h2) + h2, torch.arange(h2)] = -1.0
        P[torch.arange(h2), torch.arange(h2) + h2] = 1.0
        self.register_buffer("P", P, persistent=False)

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def rms(self, x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

    def forward(self, input_ids, attention_mask):
        bb = self.m.backbone
        B, L = input_ids.shape
        h = bb.embed_tokens(input_ids)
        am = attention_mask.to(h.dtype)
        full = (1.0 - am)[:, None, None, :] * -1e4
        idx = torch.arange(L, device=h.device)
        dist = (idx[None, :] - idx[:, None]).abs()
        masks = {None: full}
        for w in set(x for x in self.windows if x is not None):
            masks[w] = full + ((dist > w).to(h.dtype) * -1e4)[None, None]
        for li, layer in enumerate(bb.layers):
            nm = self.rope_names[self.thetas[li]]
            cos, sin = getattr(self, f"cos_{nm}")[:L], getattr(self, f"sin_{nm}")[:L]
            x = self.rms(h, layer.input_layernorm.weight)
            at = layer.self_attn
            q = at.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
            k = at.k_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
            v = at.v_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
            if ROPE_MODE == "slice":
                q = q * cos + self._rot(q) * sin
                k = k * cos + self._rot(k) * sin
            else:
                q = q * cos + torch.matmul(q, self.P) * sin
                k = k * cos + torch.matmul(k, self.P) * sin
            a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * (self.hd ** -0.5) + masks[self.windows[li]], -1)
            o = torch.matmul(a, v).transpose(1, 2).reshape(B, L, -1)
            h = h + at.o_proj(o)
            y = self.rms(h, layer.post_attention_layernorm.weight)
            mlp = layer.mlp
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(y)) * mlp.up_proj(y))
        h = self.rms(h, bb.norm.weight)
        if self.m.pooling == "cls":
            pooled = h[:, 0]
        else:
            mm = am.unsqueeze(-1)
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1.0)
        return self.m.probs(self.m.head(pooled))


def make_lean(m: DecisionModel):
    return LeanUltraBert(m) if hasattr(m.backbone, "embed_tokens") else LeanModernBert(m)


@torch.no_grad()
def smooth_mlp(w: "LeanModernBert", tok, texts, alpha: float = 0.5, max_len: int = 384):
    """SmoothQuant for the MLP down-projection: its input act(in)*gate has large per-channel outliers that break
    int8 activation quantization. Divide channel j of the input by s_j (folded into the gate projection rows,
    exact because the product is linear in gate) and multiply column j of Wo by s_j."""
    layers = w.m.backbone.layers
    ub = isinstance(w, LeanUltraBert)
    down = (lambda l: l.mlp.down_proj) if ub else (lambda l: l.mlp.Wo)
    amax = [None] * len(layers)
    hooks = []
    for i, l in enumerate(layers):
        def hk(mod, inp, i=i):
            a = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0)
            amax[i] = a if amax[i] is None else torch.maximum(amax[i], a)
        hooks.append(down(l).register_forward_pre_hook(hk))
    for s in range(0, len(texts), 16):
        enc = tok(texts[s:s + 16], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        w(enc["input_ids"], enc["attention_mask"])
    for h in hooks:
        h.remove()
    for i, l in enumerate(layers):
        wmax = down(l).weight.abs().amax(0).clamp(min=1e-5)
        sc = (amax[i].clamp(min=1e-5) ** alpha / wmax ** (1 - alpha)).clamp(min=1e-5)
        lin = l.mlp.up_proj if ub else w.wi_gate[i]  # the factor the down-proj input is linear in
        lin.weight.div_(sc[:, None])
        if lin.bias is not None:
            lin.bias.div_(sc)
        down(l).weight.mul_(sc[None, :])


def calib_texts(n: int = 256):
    import random
    from benchmarks.v2 import protocol
    from ownyourjev.spec import render_state
    pool = protocol.pool("boolq")
    random.Random(123).shuffle(pool)
    return [render_state(e.state) for e in pool[:n]]


def lean_export(model_dir, out_path=None, check: bool = True, int8: bool = True, smooth: float | None = None):
    from transformers import AutoTokenizer
    from ownyourjev.export import quantize_int8
    model_dir = Path(model_dir)
    out_path = Path(out_path) if out_path else model_dir / "model.onnx"
    m = DecisionModel.load(model_dir, device="cpu").float().eval()
    w = make_lean(m).eval()
    tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
    if smooth is not None:
        smooth_mlp(w, tok, calib_texts(), alpha=smooth)
    if check:
        long = " ".join(["the quick brown fox jumps over the lazy dog"] * 40)
        enc = tok([long, "short text here"], padding=True, truncation=True, max_length=384, return_tensors="pt")
        with torch.no_grad():
            ref = m.probs(m(enc["input_ids"], enc["attention_mask"]))
            got = w(enc["input_ids"], enc["attention_mask"])
        print("lean vs HF max abs diff:", float((ref - got).abs().max()), ref.tolist(), got.tolist())
    enc = tok(["example input for tracing", "second"], padding=True, return_tensors="pt")
    torch.onnx.export(w, (enc["input_ids"], enc["attention_mask"]), str(out_path), input_names=["input_ids", "attention_mask"],
                      output_names=["probs"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    if int8:
        from quant import quantize_weights_only
        quantize_weights_only(out_path, out_path.with_suffix(".int8.onnx"))
    return out_path


if __name__ == "__main__":
    for d in sys.argv[1:]:
        lean_export(d)
