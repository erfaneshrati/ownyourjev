"""Train a gte-modernbert student on pool labels (+ the 20 gold seeds); select on dev (gold).
labels file: {uid: intent | {intent: p}}. Saves model, dev probs, and optionally ONNX int8 + its dev accuracy."""
import json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import typer
from benchmarks.v2 import protocol as P
from ownyourjev.train import TrainConfig, train_with_oom_retry, predict_probs
from ownyourjev.evaluate import evaluate

T = "banking77"
MD = Path("data/v2/banking77/models")


def build(labels_path, soft, min_conf=0.0, seeds_rep=1):
    lab = json.loads(Path(labels_path).read_text())
    q = P.task_spec(T).parsed["answer"]
    out = []
    for e in P.pool(T):
        v = lab.get(e.uid)
        if v is None:
            continue
        if isinstance(v, dict):
            k = max(v, key=v.get)
            if v[k] < min_conf:
                continue
            meta = {"soft": {"answer": [float(v.get(o, 0.0)) for o in q.options]}} if soft else {}
        else:
            k, meta = v, {}
        out.append(replace(e, labels={"answer": k}, meta={**e.meta, **meta}))
    out += P.seeds(T) * seeds_rep
    return q, out


def main(labels: str, name: str, soft: bool = False, epochs: int = 6, lr: float = 3e-5, bs: int = 32, max_len: int = 64,
         backbone: str = "gte-modernbert", method: str = "full", seed: int = 0, min_conf: float = 0.0, onnx: bool = False, ls: float = 0.0, clean: bool = False, gold_train: str = "", gold_rep: int = 1):
    q, tr = build(labels, soft, min_conf)
    dev = P.dev(T)
    A = [dev[i] for i in range(0, 200, 2)]; B = [dev[i] for i in range(1, 200, 2)]
    val = A if clean else dev  # clean: select on dev half A, report on unseen half B
    if gold_train == "A": tr += A * gold_rep
    elif gold_train == "B": tr += B * gold_rep; val = A
    elif gold_train == "all": tr += dev * gold_rep
    cfg = TrainConfig(backbone=backbone, method=method, lr=lr, epochs=epochs, batch_size=bs, max_len=max_len, seed=seed,
                      soft_labels=soft, patience=3, gpu_need_gb=5.0, label_smoothing=ls)
    t0 = time.time()
    res = train_with_oom_retry(q, tr, val, cfg)
    out = MD / name
    res.model.save(out, res.tokenizer)
    pd = predict_probs(res.model, res.tokenizer, dev)
    gold = [e.labels["answer"] for e in dev]
    m = evaluate(q, gold, pd)
    np.save(out / "dev_probs.npy", pd)
    rec = {"name": name, "labels": labels, "n_train": len(tr), "soft": soft, "epochs": epochs, "lr": lr, "best_epoch": res.best_epoch,
           "dev_acc": m["accuracy"], "devA_acc": evaluate(q, gold[0::2], pd[0::2])["accuracy"], "devB_acc": evaluate(q, gold[1::2], pd[1::2])["accuracy"], "history": res.history, "wall_s": time.time() - t0}
    if onnx:
        from ownyourjev.export import export_onnx, quantize_int8, OnnxPredictor
        export_onnx(out); quantize_int8(out / "model.onnx")
        p8 = OnnxPredictor(out, threads=4).probs([e.state for e in dev])
        rec["dev_acc_int8"] = evaluate(q, gold, p8)["accuracy"]
    (out / "rec.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "history"}, default=float))

if __name__ == "__main__":
    typer.run(main)
