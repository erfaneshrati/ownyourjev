"""Synthetic-data showcase: train on examples generated from the question spec alone.

Run after the main benchmark so the recipe selected on silver labels is available:
    uv run python -m benchmarks.synthetic prompt_injections --n 1000
Writes benchmarks/results/<task>/synthetic.json with two rows: pure synthetic (spec only) and synthetic
guided by 20 real examples shown to the generator (never trained on).
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import numpy as np
import torch
import typer
from rich import print as rprint

from benchmarks.run import ci_eval, eval_on, overlap, jev_answers, jev_probs_matrix, with_gold_labels
from benchmarks.tasks import get_task
from ownyourjev.data import read_jsonl, train_val_split, write_jsonl
from ownyourjev.llm import Budget, OpenRouterClient
from ownyourjev.synth import synthesize
from ownyourjev.train import TrainConfig, train_with_oom_retry

app = typer.Typer()


def generate(task_name: str, n: int, domain: str, seeds, path: Path) -> list:
    if path.is_file():
        return read_jsonl(path)
    task = get_task(task_name)
    questions = task.spec.parsed
    fields = list(task.eval()[0].state.keys()) if isinstance(task.eval()[0].state, dict) else None
    client = OpenRouterClient(budget=Budget(), concurrency=16)

    async def _go():
        try:
            return await synthesize(client, questions, n, purpose=f"synth:{task_name}", domain=domain, state_fields=fields, seeds=seeds, batch_size=8)
        finally:
            await client.aclose()

    out = asyncio.run(_go())
    write_jsonl(path, out)
    rprint(f"  generated {len(out)} (asked {n}); spent ${client.stats['cost']:.4f}")
    return out


@app.command()
def main(task_name: str, n: int = 1000, domain: str = "user messages and prompts sent to AI assistants and chatbots, in English and German", results: Path = Path("benchmarks/results")):
    task = get_task(task_name)
    spec = task.spec
    head = spec.primary_head
    q = spec.parsed[head]
    res_path = results / task_name / "result.json"
    base = json.loads(res_path.read_text())
    recipe = TrainConfig(**base["variants"]["silver"]["recipe"])
    evalset = task.eval()
    pool = task.pool()
    seeds = with_gold_labels(random.Random(0).sample(pool, 20))
    data_dir = Path("data/silver") / task_name
    jev = jev_answers(spec.questions, evalset, head)
    jev_m = jev_probs_matrix(q, jev, evalset) if jev else None
    out = {}
    for variant, sd in [("synthetic", None), ("synthetic_guided", seeds)]:
        data = generate(task_name, n, domain, sd, data_dir / f"{variant}.jsonl")
        data = [e for e in data if e.valid and head in e.labels]
        tr, va = train_val_split(data, 0.1, seed=0)
        res = train_with_oom_retry(q, tr, va, recipe, qid=head)
        if res is None:
            raise RuntimeError('out of memory on every device')
        probs, _ = eval_on(q, res, evalset, head, None)
        m = ci_eval(q, evalset, head, probs, None, spec.primary_metric)
        out[variant] = {"n_train": len(tr), "label_balance": float(np.mean([e.labels[head] for e in data])) if q.type == "noul" else None, "final": m, "recipe": recipe.to_dict(), "gpu_s": res.train_s, "overlap_with_jev": overlap(q, [e.gold[head] for e in evalset], probs, jev_m)}
        rprint(f"  {variant}: {spec.primary_metric}={m['primary_value']:.4f} CI {m['ci95']} (n_train={len(tr)}) vs Jev {base['jev_published'].get('jev')}")
        del res
        torch.cuda.empty_cache()
    (results / task_name / "synthetic.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    app()
