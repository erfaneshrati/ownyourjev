"""Shared-encoder multi-head scorer for HelpSteer2 (one gte-modernbert pass -> 5 score heads), with
response-preserving truncation and a length header, ONNX export and an onnxruntime predictor."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HF_ID = "Alibaba-NLP/gte-modernbert-base"
A = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
K = 5


def truncate_layers(bb, n):
    if n and n < bb.config.num_hidden_layers:
        bb.layers = bb.layers[:n]
        bb.config.num_hidden_layers = n
        if getattr(bb.config, "layer_types", None):
            bb.config.layer_types = bb.config.layer_types[:n]
    return bb


class Encoder:
    """state -> token ids: [CLS] header [SEP] prompt(head+tail) [SEP] response(head+tail) [SEP], at most `budget` ids."""

    def __init__(self, tok, budget=128, p_frac=0.3, header=True, resp_tail_frac=0.33, prompt_tail_frac=0.67):
        self.tok, self.budget, self.p_frac, self.header = tok, budget, p_frac, header
        self.rt, self.pt = resp_tail_frac, prompt_tail_frac
        self.cls, self.sep = tok.cls_token_id, tok.sep_token_id

    @staticmethod
    def _cut(ids, n, tail_frac):
        if len(ids) <= n:
            return ids
        nt = int(round(n * tail_frac)); nh = n - nt
        return ids[:nh] + (ids[len(ids) - nt:] if nt else [])

    def ids(self, state):
        p = self.tok(state["prompt"], add_special_tokens=False)["input_ids"]
        r = self.tok(state["response"], add_special_tokens=False)["input_ids"]
        h = []
        if self.header:
            nl = state["response"].count("\n") + 1
            h = self.tok(f"prompt {len(p)} response {len(r)} lines {nl}", add_special_tokens=False)["input_ids"] + [self.sep]
        room = self.budget - len(h) - 3
        pt = min(len(p), max(int(room * self.p_frac), room - len(r)))
        rt = min(len(r), room - pt)
        return [self.cls] + h + self._cut(p, pt, self.pt) + [self.sep] + self._cut(r, rt, self.rt) + [self.sep]

    def config(self):
        return dict(budget=self.budget, p_frac=self.p_frac, header=self.header, resp_tail_frac=self.rt, prompt_tail_frac=self.pt)


class MultiHead(nn.Module):
    def __init__(self, backbone, pooling="cls", dropout=0.1, n_heads=len(A)):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        hdim = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.Linear(hdim, n_heads * K)
        self.n_heads = n_heads
        self.register_buffer("temperature", torch.ones(n_heads))

    def embed(self, ids, mask):
        hs = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            return hs[:, 0]
        m = mask.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, mask):
        return self.heads(self.dropout(self.embed(ids, mask))).view(-1, self.n_heads, K)

    def probs(self, logits, calibrated=True):
        t = self.temperature.view(1, -1, 1) if calibrated else 1.0
        return torch.softmax(logits.float() / t, -1)


def build(n_layers=None, pooling="cls", dropout=0.1):
    bb = AutoModel.from_pretrained(HF_ID, dtype=torch.float32)
    truncate_layers(bb, n_layers)
    return MultiHead(bb, pooling, dropout), AutoTokenizer.from_pretrained(HF_ID)


def save(model, tok, enc: Encoder, path, extra=None):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path / "backbone"); tok.save_pretrained(path / "backbone")
    torch.save({"heads": model.heads.state_dict(), "temperature": model.temperature.cpu()}, path / "heads.pt")
    (path / "meta.json").write_text(json.dumps({"attributes": A, "pooling": model.pooling, "n_heads": model.n_heads, "n_layers": model.backbone.config.num_hidden_layers,
                                                "encoder": enc.config(), **(extra or {})}, indent=1))


def load(path, device="cpu"):
    path = Path(path); meta = json.loads((path / "meta.json").read_text())
    bb = AutoModel.from_pretrained(path / "backbone", dtype=torch.float32)
    m = MultiHead(bb, meta["pooling"], n_heads=meta.get("n_heads", len(A)))
    st = torch.load(path / "heads.pt", map_location="cpu")
    m.heads.load_state_dict(st["heads"]); m.temperature.copy_(st["temperature"])
    tok = AutoTokenizer.from_pretrained(path / "backbone")
    return m.to(device).eval(), tok, Encoder(tok, **meta["encoder"]), meta


class _W(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, ids, mask):
        return self.m.probs(self.m(ids, mask))


def export(path, int8=True, keep_wo_layers=None, out_name="model.int8.onnx"):
    """keep_wo_layers: layers whose mlp.Wo stays fp32 (None = all layers)."""
    path = Path(path)
    if not (path / "model.onnx").exists():
        m, tok, enc, _ = load(path)
        ids = torch.randint(1000, 2000, (2, 16)); mask = torch.ones_like(ids)
        torch.onnx.export(_W(m), (ids, mask), str(path / "model.onnx"), input_names=["input_ids", "attention_mask"], output_names=["probs"],
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "probs": {0: "b"}}, opset_version=17, dynamo=False)
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx
        g = onnx.load(str(path / "model.onnx"), load_external_data=False)
        # keep mlp.Wo (activation outliers: int8 there cost ~0.04 dev Spearman) and the tiny head in fp32
        def keep(nm):
            if "heads" in nm: return True
            if "mlp/Wo" not in nm: return False
            return keep_wo_layers is None or any(f"layers.{i}/" in nm for i in keep_wo_layers)
        excl = [n.name for n in g.graph.node if n.op_type in ("MatMul", "Gemm") and keep(n.name)]
        quantize_dynamic(str(path / "model.onnx"), str(path / out_name), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl)


class OnnxMH:
    """End-to-end predictor: state -> (5, 5) probabilities (tokenisation + truncation + forward + softmax)."""

    def __init__(self, path, onnx_name="model.int8.onnx", threads=4):
        import onnxruntime as ort
        path = Path(path); meta = json.loads((path / "meta.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(path / "backbone")
        self.enc = Encoder(self.tok, **meta["encoder"])
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path / onnx_name), o, providers=["CPUExecutionProvider"])

    def probs(self, state):
        ids = np.asarray([self.enc.ids(state)], dtype=np.int64)
        return self.sess.run(["probs"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0][:len(A)]  # heads 0-4 (flash/gold heads)


def optimize_and_quantize(path, keep_wo_layers=None, out_name="model.opt.int8.onnx"):
    """ORT transformer graph fusions (SkipLayerNorm, Gelu; halves the node count -> less per-op overhead at batch 1),
    then the same dynamic int8 quantisation as export() (mlp.Wo of keep_wo_layers and the head stay fp32)."""
    from onnxruntime.transformers import optimizer
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx
    path = Path(path)
    if not (path / "model.onnx").exists():
        export(path, int8=False)
    m = optimizer.optimize_model(str(path / "model.onnx"), model_type="bert", num_heads=12, hidden_size=768, opt_level=1)
    m.save_model_to_file(str(path / "model.opt.onnx"))
    g = onnx.load(str(path / "model.opt.onnx"), load_external_data=False)
    def keep(nm):
        if "heads" in nm: return True
        if "mlp/Wo" not in nm: return False
        return keep_wo_layers is None or any(f"layers.{i}/" in nm for i in keep_wo_layers)
    excl = [n.name for n in g.graph.node if n.op_type in ("MatMul", "Gemm") and keep(n.name)]
    assert len(excl) >= 1 + (len(keep_wo_layers) if keep_wo_layers is not None else 1), excl
    quantize_dynamic(str(path / "model.opt.onnx"), str(path / out_name), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excl, extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    return len(excl)
