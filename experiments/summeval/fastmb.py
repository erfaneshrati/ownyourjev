"""Export-friendly ModernBERT forward (same weights, same math) for batch CPU inference via ONNX:
rotary tables precomputed as constants, rotate_half as a constant 64x64 matmul, one additive window mask built once,
plain eager attention. Removes most of the small Slice/Concat/Where ops that dominate the stock export on CPU."""
import math
import torch
from torch import nn


class FastEncoder(nn.Module):
    def __init__(self, mb, max_len: int):
        super().__init__()
        c = mb.config
        self.mb = mb
        self.H = c.num_attention_heads
        self.dh = c.hidden_size // self.H
        self.window = c.sliding_window  # |i-j| <= window
        self.types = [l.attention_type for l in mb.layers]
        pos = torch.arange(max_len).unsqueeze(0)
        x = torch.zeros(1, max_len, c.hidden_size)
        tabs = {}
        for t in set(self.types):
            cos, sin = mb.rotary_emb(x, pos, t)
            tabs[t] = (cos[0].float(), sin[0].float())  # (L, dh)
        self.register_buffer("cos_g", tabs.get("full_attention", tabs[self.types[0]])[0])
        self.register_buffer("sin_g", tabs.get("full_attention", tabs[self.types[0]])[1])
        self.register_buffer("cos_l", tabs.get("sliding_attention", tabs[self.types[0]])[0])
        self.register_buffer("sin_l", tabs.get("sliding_attention", tabs[self.types[0]])[1])
        R = torch.zeros(self.dh, self.dh)
        h = self.dh // 2
        for i in range(h):
            R[i + h, i] = -1.0  # out[:, i] = -x[:, i+h]
            R[i, i + h] = 1.0   # out[:, i+h] = x[:, i]
        self.register_buffer("R", R)

    def forward(self, input_ids, attention_mask):
        mb = self.mb
        B, L = input_ids.shape
        x = mb.embeddings(input_ids=input_ids)
        idx = torch.arange(L, device=input_ids.device)
        pad = (1.0 - attention_mask.to(x.dtype))[:, None, None, :] * -1e9  # (B,1,1,L)
        far = ((idx[None, :] - idx[:, None]).abs() > self.window).to(x.dtype) * -1e9  # (L,L)
        m_g = pad
        m_l = pad + far[None, None]
        cg, sg = self.cos_g[:L][None, None], self.sin_g[:L][None, None]
        cl, sl = self.cos_l[:L][None, None], self.sin_l[:L][None, None]
        scale = self.dh ** -0.5
        for layer, t in zip(mb.layers, self.types):
            hdn = layer.attn_norm(x)
            qkv = layer.attn.Wqkv(hdn).view(B, L, 3, self.H, self.dh)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,L,dh)
            if t == "sliding_attention":
                c, s, m = cl, sl, m_l
            else:
                c, s, m = cg, sg, m_g
            q = q * c + (q @ self.R) * s
            k = k * c + (k @ self.R) * s
            a = torch.softmax((q @ k.transpose(-1, -2)) * scale + m, dim=-1)
            o = (a @ v).transpose(1, 2).reshape(B, L, -1)
            x = x + layer.attn.Wo(o)
            hm = layer.mlp_norm(x)
            inp, gate = layer.mlp.Wi(hm).chunk(2, dim=-1)
            x = x + layer.mlp.Wo(layer.mlp.act(inp) * gate)
        return mb.final_norm(x)


class FastMH(nn.Module):
    def __init__(self, m, max_len):
        super().__init__()
        self.enc = FastEncoder(m.backbone, max_len)
        self.m = m

    def forward(self, ids, mask, feats):
        h = self.m.pool_h(self.enc(ids, mask), mask)
        z = self.m.head(h)
        if self.m.fhead is not None:
            z = z + self.m.fhead((feats - self.m.f_mu) / self.m.f_sd)
        z = z.view(-1, len(self.m.cfg.dims), 5)
        return torch.softmax(z / self.m.temp.view(1, -1, 1), -1)
