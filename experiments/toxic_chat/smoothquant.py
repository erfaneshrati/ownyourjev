"""SmoothQuant for ModernBERT mlp/Wo: migrate per-channel activation scale of the GeGLU output into Wo
(gate half of Wi divided by s, Wo columns multiplied by s; exact in fp32), so dynamic int8 can quantise Wo too.
usage: smoothquant.py SRC_MODEL DST_MODEL [alpha]"""
import sys, random, shutil, numpy as np, torch
from common import *
from mh import MH, texts_of, export_onnx, onnx_probs
from quant import quantize_selective
src, dst = sys.argv[1], sys.argv[2]; alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
m, tok = MH.load(D / "models" / src, attn="eager")
pool = protocol.pool(T); random.Random(0).shuffle(pool)
cal = texts_of(pool[:384])
amax = {}
hooks = []
for i, layer in enumerate(m.backbone.layers):
    def hk(mod, inp, i=i):
        a = inp[0].detach().abs().amax(dim=(0, 1))
        amax[i] = torch.maximum(amax[i], a) if i in amax else a
    hooks.append(layer.mlp.Wo.register_forward_pre_hook(hk))
with torch.no_grad():
    for t in cal:
        enc = tok([t], truncation=True, max_length=m.max_len, return_tensors="pt")
        m(enc["input_ids"], enc["attention_mask"])
for h in hooks: h.remove()
with torch.no_grad():
    for i, layer in enumerate(m.backbone.layers):
        Wo, Wi = layer.mlp.Wo, layer.mlp.Wi
        I = Wo.weight.shape[1]
        wmax = Wo.weight.abs().amax(dim=0).clamp(min=1e-5)
        s = (amax[i].clamp(min=1e-5) ** alpha / wmax ** (1 - alpha)).clamp(min=1e-2, max=1e3)
        Wi.weight[I:] /= s[:, None]
        if Wi.bias is not None: Wi.bias[I:] /= s
        Wo.weight *= s[None, :]
        if i in (0, 10, 21): print(f"layer {i}: act max {amax[i].max():.1f} median {amax[i].median():.2f}; s range {s.min():.3f}-{s.max():.2f}")
m.save(D / "models" / dst, tok)
md = D / "models" / dst
export_onnx(md, int8=False, attn="eager")
quantize_selective(md / "model.onnx", md / "model.int8.onnx", "A")          # full int8 incl. mlp/Wo
quantize_selective(md / "model.onnx", md / "model.int8AB.onnx", "AB")       # mlp/Wo fp32, for reference
dev = protocol.dev(T)
ref = np.load(D / "runs" / src / "dev_probs.npy")
(D / "runs" / dst).mkdir(parents=True, exist_ok=True)
for f in ["model.int8.onnx", "model.int8AB.onnx"]:
    p = onnx_probs(md, texts_of(dev), f, threads=4)
    np.save(D / "runs" / dst / f"dev_probs_{f.split('.')[1]}.npy", p)
    print(f, "max|diff| vs torch src", np.abs(p - ref).max(0).round(3), "mean", np.abs(p - ref).mean(0).round(4))
    for i, h in enumerate(HEADS):
        print("  ", report([e.gold[h] for e in dev], p[:, i], h))
