"""Train a (possibly layer-truncated) gte-modernbert student for boolq and evaluate on dev.

Layer truncation is done by wrapping ownyourjev.train.load_backbone (no library edits): keep the first N
encoder layers of the pretrained backbone (or load a given model dir's backbone to continue from it).
"""
from __future__ import annotations
import argparse, json, random, time
import numpy as np, torch
from common import *
import ownyourjev.train as T
from ownyourjev.models import decision as D
from ownyourjev.evaluate import binary_metrics
from ownyourjev.spec import Question

_orig_load = D.load_backbone


TRUNC_LEFT = False
DROP: list[int] = []


def make_loader(n_layers: int | None, init_dir: str | None):
    def load(spec, dtype=torch.float32):
        if init_dir:
            from transformers import AutoModel, AutoTokenizer
            bb = AutoModel.from_pretrained(Path(init_dir) / "backbone", dtype=dtype)
            tok = AutoTokenizer.from_pretrained(Path(init_dir) / "backbone")
        else:
            bb, tok = _orig_load(spec, dtype)
        if TRUNC_LEFT:
            tok.truncation_side = "left"  # keep the question (it comes after the passage)
        if DROP or (n_layers and n_layers < bb.config.num_hidden_layers):
            keep = [i for i in range(bb.config.num_hidden_layers) if i not in DROP] if DROP else layer_plan(bb.config.num_hidden_layers, n_layers)
            bb.layers = torch.nn.ModuleList([bb.layers[i] for i in keep])
            bb.config.layer_types = [bb.config.layer_types[i] for i in keep]
            bb.config.num_hidden_layers = len(keep)
            for j, l in enumerate(bb.layers):
                l.layer_idx = j
                at = getattr(l, "attn", None) or getattr(l, "self_attn", None)
                if at is not None:
                    at.layer_idx = j
        return bb, tok
    return load


def layer_plan(total: int, n: int) -> list[int]:
    return list(range(n))  # first-N layers (keeps the full/sliding pattern)


def load_set(spec_str: str, soft_key: str | None) -> list[Example]:
    """spec_str: path[:max_n]. Examples need labels; soft probs in meta['soft']['answer'] if present."""
    p, _, n = spec_str.partition(":")
    exs = [e for e in read_jsonl(p) if e.labels is not None]
    gu = gold_uids()
    exs = [e for e in exs if e.uid not in gu and e.meta.get("src_uid") not in gu]
    if n:
        random.Random(1).shuffle(exs); exs = exs[: int(n)]
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", default=None, help="jsonl for early stopping (default: 400 held-out from first train set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--gpu_gb", type=float, default=7.0)
    ap.add_argument("--pooling", default=None)
    ap.add_argument("--backbone", default="gte-modernbert")
    ap.add_argument("--trunc_left", action="store_true")
    ap.add_argument("--init_head", action="store_true")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--drop_layers", default="", help="comma-separated layer indices to drop")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--add_gold", action="store_true", help="also train on the 20 seeds")
    a = ap.parse_args()
    global TRUNC_LEFT, DROP
    TRUNC_LEFT = a.trunc_left
    DROP = [int(x) for x in a.drop_layers.split(",") if x]
    T.load_backbone = make_loader(a.layers or None, a.init)
    if a.init and a.init_head:  # shared init: also start from the init model's head (and temperature reset)
        _ob = T._build_model
        def _build(question, cfg, _ob=_ob, d=a.init):
            m, tok = _ob(question, cfg)
            st = torch.load(Path(d) / "head.pt", map_location="cpu")
            m.head.load_state_dict(st["head"])
            return m.to(cfg.device), tok
        T._build_model = _build
    tr = []
    for s in a.train:
        tr += load_set(s, None)
    if a.val:
        va = load_set(a.val, None)
        vu = {e.uid for e in va}
        tr = [e for e in tr if e.uid not in vu and e.meta.get("src_uid") not in {v.uid for v in va}]
    else:
        first = [e for e in tr if not e.meta.get("synthetic")]
        random.Random(1).shuffle(first)
        va = first[:400]; vu = {e.uid for e in va}
        tr = [e for e in tr if e.uid not in vu]
    if a.add_gold:
        tr += protocol.seeds(TASK)
    random.Random(a.seed).shuffle(tr)
    print(f"train={len(tr)} val={len(va)} yes_rate={np.mean([e.labels[QID] for e in tr]):.3f}", flush=True)
    cfg = T.TrainConfig(backbone=a.backbone, method="full", lr=a.lr, epochs=a.epochs, batch_size=a.bs, max_len=a.max_len,
                        seed=a.seed, patience=a.patience, soft_labels=a.soft, pooling=a.pooling, warmup_frac=a.warmup, weight_decay=a.wd, label_smoothing=a.ls, gpu_need_gb=a.gpu_gb, eval_batch_size=32)
    t0 = time.time()
    res = T.train_with_oom_retry(Q[QID], tr, va, cfg)
    m, tok = res.model, res.tokenizer
    dev = protocol.dev(TASK)
    p = T.predict_probs(m, tok, dev, batch_size=32)
    g = [e.gold[QID] for e in dev]
    bm = binary_metrics(g, p)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m.save(out, tok)
    np.save(out / "dev_probs.npy", p)
    rec = {"args": vars(a), "n_train": len(tr), "dev_acc": bm["accuracy"], "dev_acc_A": float(np.mean((p[:100] >= .5) == np.array(g[:100]))),
           "dev_acc_B": float(np.mean((p[100:] >= .5) == np.array(g[100:]))), "dev_auroc": bm["auroc"], "dev_nll": bm["nll"],
           "history": res.history, "best_epoch": res.best_epoch, "temperature": res.temperature, "wall_s": time.time() - t0}
    (out / "result.json").write_text(json.dumps(rec, indent=1, default=float))
    with open(RES / "runs.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "history"}, default=float) + "\n")
    print("RESULT", json.dumps({k: rec[k] for k in ["n_train", "dev_acc", "dev_acc_A", "dev_acc_B", "dev_auroc", "best_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
