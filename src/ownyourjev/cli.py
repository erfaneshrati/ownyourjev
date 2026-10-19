"""ownyourjev command line."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from dotenv import find_dotenv, load_dotenv
from rich import print as rprint
from rich.table import Table

load_dotenv(find_dotenv(usecwd=True))
app = typer.Typer(help="Train your own Jev-style decision model (Noul / Choice / Score).", no_args_is_help=True)


def _load_questions(spec_path: Path) -> dict:
    from ownyourjev.spec import Question

    raw = json.loads(Path(spec_path).read_text())
    qs = raw.get("questions", raw)
    return {qid: Question(**q) for qid, q in qs.items()}


def _read_states(path: Path) -> list:
    """Unlabeled inputs: .jsonl of {"state": ...} / {"uid", "state"} or .txt with one state per line."""
    from ownyourjev.data import Example

    if path.suffix == ".jsonl":
        out = []
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            if "state" in d:
                out.append(Example(str(d.get("uid", i)), d["state"], d.get("labels"), d.get("gold"), d.get("group", ""), d.get("meta", {})))
            else:
                out.append(Example(str(i), d))
        return out
    return [Example(str(i), line.strip()) for i, line in enumerate(path.read_text().splitlines()) if line.strip()]


@app.command()
def label(
    spec: Path = typer.Argument(..., help="JSON file with {\"questions\": {qid: {type, instructions, criteria}}}"),
    inputs: Path = typer.Argument(..., help=".jsonl of {\"state\": ...} or .txt one state per line"),
    out: Path = typer.Option(Path("data/silver.jsonl"), help="output jsonl"),
    model: Optional[str] = typer.Option(None, help="OpenRouter model id (default OYJ_LABELER_MODEL)"),
    seeds: Optional[Path] = typer.Option(None, help="jsonl of labeled examples to show as few-shot seeds"),
    guidance: Optional[Path] = typer.Option(None, help="text file with labeling guidelines (write them from your seeds and the labeler's dev errors)"),
    soft: bool = typer.Option(False, "--soft", help="ask for a probability per option (soft labels; train with soft_labels=True)"),
    limit: Optional[int] = typer.Option(None),
    max_cost: Optional[float] = typer.Option(None, help="abort if the pre-flight estimate exceeds this (USD)"),
    concurrency: int = typer.Option(16),
):
    """Label unlabeled states with an LLM against the question spec."""
    from ownyourjev.data import read_jsonl, write_jsonl
    from ownyourjev.label import build_messages, label_examples
    from ownyourjev.llm import Budget, OpenRouterClient

    qs = _load_questions(spec)
    exs = _read_states(inputs)[: limit or None]
    seed_ex = read_jsonl(seeds) if seeds else None
    client = OpenRouterClient(model, concurrency=concurrency, budget=Budget())
    guide = guidance.read_text() if guidance else None
    est = sum(client.estimate_cost(build_messages(qs, ex.state, seeds=seed_ex, guidance=guide, distribution=soft), 64 + 24 * len(qs)) for ex in exs)
    rprint(f"[bold]{len(exs)}[/bold] states x {len(qs)} questions with {client.model}; pre-flight estimate ≈ ${est:.4f} (cache hits are free)")
    if max_cost is not None and est > max_cost:
        raise typer.Exit(code=2)
    async def _go():
        try:
            return await label_examples(client, qs, exs, purpose=f"label:{out.stem}", seeds=seed_ex, guidance=guide, distribution=soft)
        finally:
            await client.aclose()

    out_ex = asyncio.run(_go())
    n_ok = sum(e.valid for e in out_ex)
    write_jsonl(out, out_ex)
    rprint(f"labeled {n_ok}/{len(out_ex)} -> {out}; spent ${client.stats['cost']:.4f} ({client.stats['cached']} cached)")


@app.command()
def synth(
    spec: Path,
    out: Path = typer.Option(Path("data/synthetic.jsonl")),
    n: int = typer.Option(500),
    domain: str = typer.Option("general", help="short description of the domain of the inputs"),
    state_fields: Optional[str] = typer.Option(None, help="comma-separated field names if the state is an object"),
    seeds: Optional[Path] = typer.Option(None, help="jsonl of real labeled examples to imitate"),
    model: Optional[str] = typer.Option(None),
):
    """Generate synthetic labeled examples from the spec alone."""
    from ownyourjev.data import read_jsonl, write_jsonl
    from ownyourjev.llm import Budget, OpenRouterClient
    from ownyourjev.synth import synthesize

    qs = _load_questions(spec)
    client = OpenRouterClient(model, budget=Budget())
    fields = [f.strip() for f in state_fields.split(",")] if state_fields else None
    async def _go():
        try:
            return await synthesize(client, qs, n, purpose=f"synth:{out.stem}", domain=domain, state_fields=fields, seeds=read_jsonl(seeds) if seeds else None)
        finally:
            await client.aclose()

    exs = asyncio.run(_go())
    write_jsonl(out, exs)
    rprint(f"generated {len(exs)} -> {out}; spent ${client.stats['cost']:.4f}")


@app.command()
def train(
    spec: Path,
    data: Path = typer.Argument(..., help="labeled jsonl (from `label` or `synth`)"),
    out: Path = typer.Option(Path("models/my-decision")),
    qid: Optional[str] = typer.Option(None, help="question id to train (default: first)"),
    backbone: str = typer.Option("ultrabert"),
    method: str = typer.Option("full", help="frozen | lora | full"),
    epochs: int = typer.Option(5),
    lr: Optional[float] = typer.Option(None),
    batch_size: int = typer.Option(16),
    max_len: int = typer.Option(512),
    val_frac: float = typer.Option(0.1),
    seed: int = typer.Option(0),
    soft: bool = typer.Option(False, "--soft", help="train on soft labels from `label --soft`"),
):
    """Train one decision model for one question."""
    from ownyourjev.data import read_jsonl, train_val_split
    from ownyourjev.train import TrainConfig
    from ownyourjev.train import train as _train

    qs = _load_questions(spec)
    qid = qid or next(iter(qs))
    exs = [e for e in read_jsonl(data) if e.valid and qid in e.labels]
    tr, va = train_val_split(exs, val_frac, seed=seed)
    cfg = TrainConfig(backbone=backbone, method=method, epochs=epochs, lr=lr, batch_size=batch_size, max_len=max_len, seed=seed, soft_labels=soft)
    res = _train(qs[qid], tr, va, cfg, qid=qid, verbose=True)
    res.model.save(out, res.tokenizer)
    rprint(res.summary())
    rprint(f"saved -> {out}")


@app.command()
def search(
    spec: Path,
    data: Path,
    out: Path = typer.Option(Path("runs/search")),
    qid: Optional[str] = typer.Option(None),
    backbones: str = typer.Option("ultrabert,ultrabert-embed,gte-modernbert"),
    methods: str = typer.Option("frozen,lora,full"),
    stages: str = typer.Option("ABCF"),
    hpo_trials: int = typer.Option(10),
    hpo_gpu_minutes: float = typer.Option(20.0),
    max_len: int = typer.Option(512),
    val_frac: float = typer.Option(0.1),
):
    """Search backbone x method x data size x hyperparameters; writes search.jsonl, report.json, best_model/."""
    from ownyourjev.data import read_jsonl, train_val_split
    from ownyourjev.search import Search, SearchConfig

    qs = _load_questions(spec)
    qid = qid or next(iter(qs))
    exs = [e for e in read_jsonl(data) if e.valid and qid in e.labels]
    tr, va = train_val_split(exs, val_frac)
    cfg = SearchConfig(backbones=backbones.split(","), methods=methods.split(","), hpo_trials=hpo_trials, hpo_gpu_minutes=hpo_gpu_minutes, max_len=max_len)
    rep = Search(qs[qid], qid, tr, va, cfg, out).run(stages)
    rprint({k: rep[k] for k in ("selected", "gpu_seconds_total", "n_trials")})


@app.command()
def predict(model_dir: Path, inputs: Path, out: Optional[Path] = typer.Option(None), batch_size: int = typer.Option(64)):
    """Run a saved model over states and print Jev-style answers."""
    import torch
    from transformers import AutoTokenizer

    from ownyourjev.models.decision import DecisionModel
    from ownyourjev.train import pick_device, predict_probs

    m = DecisionModel.load(model_dir, device=pick_device(1.0))
    tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
    exs = _read_states(inputs)
    probs = predict_probs(m, tok, exs, batch_size=batch_size)
    rows = [{"uid": e.uid, "answer": m.answer(torch.as_tensor(p)).model_dump()} for e, p in zip(exs, probs)]
    if out:
        out.write_text("\n".join(json.dumps(r) for r in rows))
        rprint(f"wrote {len(rows)} -> {out}")
    else:
        for r in rows[:50]:
            rprint(r)


@app.command()
def calibrate(
    model_dir: Path,
    dev: Path = typer.Argument(..., help="held-out labeled .jsonl: {\"state\": ..., \"gold\": {qid: answer}} (e.g. your 200 dev examples)"),
    qid: Optional[str] = typer.Option(None, help="question id in `gold` (default: the only one)"),
):
    """Fit a post-hoc calibrator on held-out labeled examples and save it next to the model.

    Picks none / temperature / Platt / isotonic (Noul) or none / temperature (Choice, Score; never changes an
    answer) by 5-fold cross-validated log-loss. `predict`, `serve` and the ONNX predictor apply it automatically.
    Use examples the model was not trained on. Predictions are made one request at a time, as served."""
    import numpy as np

    from ownyourjev.calibration import fit_for_question

    exs = [e for e in _read_states(dev) if e.gold]
    if not exs:
        raise typer.BadParameter("no examples with a `gold` field")
    key = qid or next(iter(exs[0].gold))
    exs = [e for e in exs if e.gold.get(key) is not None]
    onnx = [p for p in (model_dir / "model.int8.onnx", model_dir / "model.onnx") if p.is_file()]
    if onnx:
        from ownyourjev.export import OnnxPredictor

        pred = OnnxPredictor(model_dir, onnx[0])
        question, probs = pred.question, np.stack([pred.probs([e.state])[0] for e in exs])
    else:
        from transformers import AutoTokenizer

        from ownyourjev.models.decision import DecisionModel
        from ownyourjev.train import predict_probs

        m = DecisionModel.load(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir / "backbone")
        question, probs = m.question, np.asarray(predict_probs(m, tok, exs, batch_size=1))
    if question.type == "noul":
        probs = probs.reshape(-1)
    cal, scores = fit_for_question(question, probs, [e.gold[key] for e in exs])
    t = Table(title=f"5-fold CV on {len(exs)} held-out examples (lower is better)")
    for c in ("method", "log-loss", "ECE", "Brier"):
        t.add_column(c)
    for name, r in scores.items():
        t.add_row(("* " if name == cal.method else "") + name, f"{r['nll']:.4f}", f"{r['ece']:.4f}", f"{r['brier']:.4f}")
    rprint(t)
    out = model_dir / "calibration.json"
    out.write_text(json.dumps({**cal.to_dict(), "n_fit": len(exs), "dev_cv": scores, "source": str(dev)}, indent=1, default=float))
    rprint(f"calibrator [bold]{cal.method}[/bold] {cal.params if cal.method != 'isotonic' else ''} -> {out}")


@app.command()
def serve(registry: Path = typer.Argument(Path("models")), host: str = "127.0.0.1", port: int = 8000):
    """Serve all models under REGISTRY with Jev's request/response schema at POST /v1/systemone."""
    import uvicorn

    from ownyourjev.serve import create_app

    uvicorn.run(create_app(registry), host=host, port=port)


@app.command()
def export(model_dir: Path, out: Optional[Path] = typer.Option(None), bench: bool = typer.Option(True), int8: bool = typer.Option(True, help="also write a dynamic-int8 model (used by default when present)")):
    """Export a saved model to ONNX (fp32 + optional int8) and measure CPU latency."""
    from ownyourjev.export import OnnxPredictor, export_onnx, quantize_int8

    p = export_onnx(model_dir, out)
    rprint(f"exported -> {p} ({p.stat().st_size / 1e6:.0f} MB)")
    if int8:
        q = quantize_int8(p)
        rprint(f"int8 -> {q} ({q.stat().st_size / 1e6:.0f} MB)")
        p = q
    if bench:
        pred = OnnxPredictor(model_dir, p)
        rprint(pred.benchmark_latency("The quick brown fox jumps over the lazy dog. " * 4))


@app.command()
def budget():
    """Show OpenRouter spend recorded by the ledger."""
    from ownyourjev.llm import Budget

    b = Budget()
    t = Table("purpose", "model", "calls", "prompt tok", "completion tok", "USD")
    for r in b.summary():
        t.add_row(r.purpose, r.model, str(r.calls), str(r.prompt_tokens), str(r.completion_tokens), f"{r.cost_usd:.4f}")
    rprint(t)
    rprint(f"[bold]spent ${b.spent():.4f} of cap ${b.cap:.2f}; remaining ${b.remaining():.4f}[/bold]")


@app.command()
def bench(
    tasks: str = typer.Argument("all", help="comma-separated task names, a mode (noul|choice|score) or 'all'"),
    variants: str = typer.Option("silver,seeded,gold", help="data variants to run"),
    pool_cap: int = typer.Option(5000),
    stages: str = typer.Option("ABCF"),
    limit: Optional[int] = typer.Option(None, help="smoke-test: cap pool and eval sizes"),
    max_cost: Optional[float] = typer.Option(None, help="abort a task if its labeling estimate exceeds this"),
    out: Path = typer.Option(Path("benchmarks/results")),
    skip_label: bool = typer.Option(False, help="only use silver labels already cached on disk"),
):
    """Run the Jev benchmark (see benchmarks/run.py)."""
    from benchmarks.run import run_benchmark

    run_benchmark(tasks, variants=variants.split(","), pool_cap=pool_cap, stages=stages, limit=limit, max_cost=max_cost, out=out, skip_label=skip_label)


@app.command()
def report(results: Path = typer.Argument(Path("benchmarks/results"))):
    """Regenerate REPORT.md and plots from benchmark results."""
    from benchmarks.report import write_report

    write_report(results)


if __name__ == "__main__":
    app()
