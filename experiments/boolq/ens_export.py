"""Relaxed setting: a K-member prediction ensemble exported as ONE ONNX graph (members share the UltraBERT tokenizer).
One onnxruntime session per request avoids the thread-pool contention of K separate sessions (whose spinning intra-op
threads fight each other: 2 sessions on 8 threads measured p95 253 ms vs 53 ms for one).
Each member: lean export + SmoothQuant(alpha) on its MLP down-projection; output = mean of member probabilities.
usage: ens_export.py OUT_DIR MEMBER_DIR... [--alpha 0.8]"""
import argparse, json, shutil
from pathlib import Path
import torch
from torch import nn
from ownyourjev.models.decision import DecisionModel
from transformers import AutoTokenizer
from fast_export import make_lean, smooth_mlp, calib_texts
from quant import quantize_weights_only


class Ens(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, input_ids, attention_mask):
        return torch.stack([m(input_ids, attention_mask) for m in self.members], 0).mean(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("members", nargs="+"); ap.add_argument("--alpha", type=float, default=0.8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(Path(a.members[0]) / "backbone")
    tok.truncation_side = "left"
    texts = calib_texts()
    leans = []
    for d in a.members:
        w = make_lean(DecisionModel.load(d).float().eval()).eval()
        smooth_mlp(w, tok, texts, alpha=a.alpha, max_len=224)
        leans.append(w)
    ens = Ens(leans).eval()
    enc = tok(["example input for tracing", "second"], padding=True, return_tensors="pt")
    torch.onnx.export(ens, (enc["input_ids"], enc["attention_mask"]), str(out / "model.onnx"), input_names=["input_ids", "attention_mask"],
                      output_names=["probs"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    quantize_weights_only(out / "model.onnx", out / "model.int8.onnx")
    (out / "model.onnx").unlink()  # keep only the int8 graph (fp32 copy is large)
    first = Path(a.members[0])
    if (out / "backbone").exists():
        shutil.rmtree(out / "backbone")
    (out / "backbone").mkdir()
    tok.save_pretrained(out / "backbone")
    meta = json.loads((first / "meta.json").read_text()); meta["max_len"] = 224; meta["ensemble_members"] = [str(Path(m).resolve()) for m in a.members]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("exported", out, len(a.members), "members")


if __name__ == "__main__":
    main()
