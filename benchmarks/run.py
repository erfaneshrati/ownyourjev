"""Benchmark orchestrator: label -> search -> evaluate -> compare with published Jev numbers.

Per task and data variant (silver | seeded | gold) the results land in ``benchmarks/results/<task>/``:
    silver.jsonl, silver_seeded.jsonl         LLM labels for the training pool (data/ dir, not committed)
    <variant>/search.jsonl, report.json       every trial of the strategy search
    result.json                               headline numbers, CIs, recipe, Jev comparison, overlap
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich import print as rprint

from benchmarks.tasks import MODES, TASKS, get_task, request_key
from ownyourjev.data import Example, kfold, read_jsonl, stratified_subsample, train_val_split, write_jsonl
from ownyourjev.evaluate import evaluate, evaluate_with_ci
from ownyourjev.label import build_messages, label_examples
from ownyourjev.llm import Budget, OpenRouterClient
from ownyourjev.search import Search, SearchConfig
from ownyourjev.spec import Question
from ownyourjev.train import TrainConfig, TrainResult, predict_probs, train_with_oom_retry

BASELINES = json.loads((Path(__file__).parent / "jev_baselines.json").read_text())
JEV_DB = Path("data/jev/responses.db")
N_SEEDS = 20


def resolve_tasks(arg: str) -> list[str]:
    if arg == "all":
        return list(TASKS)
    out: list[str] = []
    for a in arg.split(","):
        a = a.strip()
        out += MODES[a] if a in MODES else [a]
    return out


# ---- labeling --------------------------------------------------------------------------------------
def label_pool(task_name: str, questions: dict[str, Question], pool: list[Example], *, seeds: list[Example] | None, path: Path, max_cost: float | None, skip: bool, long_input: bool) -> list[Example]:
    if path.is_file():
        have = {e.uid: e for e in read_jsonl(path)}
        if all(e.uid in have for e in pool):
            return [have[e.uid] for e in pool]
    if skip:
        raise RuntimeError(f"{task_name}: silver labels missing at {path} and --skip-label set")
    client = OpenRouterClient(budget=Budget(), concurrency=96, timeout=180)
    max_tokens = 64 + 24 * len(questions)
    est = sum(client.estimate_cost(build_messages(questions, ex.state, seeds=seeds), max_tokens) for ex in pool)
    rprint(f"  labeling {len(pool)} x {len(questions)} questions with {client.model}: estimate ≈ ${est:.3f} (remaining ${client.budget.remaining():.2f})")
    if max_cost is not None and est > max_cost:
        raise RuntimeError(f"{task_name}: estimate ${est:.3f} exceeds --max-cost {max_cost}")
    t0 = time.time()
    async def _go():
        try:
            return await label_examples(client, questions, pool, purpose=f"label:{path.stem}:{task_name}", seeds=seeds, max_tokens=max_tokens)
        finally:
            await client.aclose()

    out = asyncio.run(_go())
    write_jsonl(path, out)
    rprint(f"  labeled {sum(e.valid for e in out)}/{len(out)} in {time.time() - t0:.0f}s, spent ${client.stats['cost']:.4f}")
    return out


def with_gold_labels(exs: list[Example]) -> list[Example]:
    return [Example(e.uid, e.state, dict(e.gold), e.gold, e.group, {**e.meta, "labeler": "gold"}) for e in exs]


# ---- jev per-example answers -----------------------------------------------------------------------
def jev_answers(spec_questions: dict, evalset: list[Example], head: str) -> dict[str, Any]:
    """uid -> Jev's canonical answer for ``head`` from the paper's response dump (None if absent)."""
    if not JEV_DB.is_file():
        return {}
    con = sqlite3.connect(JEV_DB)
    keys: dict[str, list[str]] = {}
    for e in evalset:  # duplicate states share a key
        keys.setdefault(request_key(e.state, spec_questions), []).append(e.uid)
    out: dict[str, Any] = {}
    ks = list(keys)
    for i in range(0, len(ks), 500):
        chunk = ks[i : i + 500]
        q = f"SELECT key, response FROM responses WHERE key IN ({','.join('?' * len(chunk))})"
        for k, resp in con.execute(q, chunk):
            a = json.loads(resp).get("answers", {}).get(head)
            if not a:
                continue
            if a["type"] == "choice":
                val = {"answer": a["choice"], "probs": a["probabilities"], "confidence": a.get("confidence")}
            elif a["type"] == "noul":
                val = {"answer": a["noul"] >= 0.5, "p": a["noul"]}
            else:
                val = {"answer": a["score"], "probs": a["probabilities"], "confidence": a.get("confidence")}
            for uid in keys[k]:
                out[uid] = val
    con.close()
    return out


def jev_probs_matrix(q: Question, jev: dict[str, Any], evalset: list[Example]) -> np.ndarray | None:
    rows = []
    for e in evalset:
        a = jev.get(e.uid)
        if a is None:
            return None
        if q.type == "noul":
            rows.append([a["p"]])
        else:
            rows.append([a["probs"].get(k, 0.0) for k in q.options])
    m = np.array(rows, dtype=float)
    return m.reshape(-1) if q.type == "noul" else m


def overlap(q: Question, golds: list, ours: np.ndarray, jev_m: np.ndarray | None) -> dict:
    if jev_m is None:
        return {}
    if q.type == "noul":
        o = (ours >= 0.5) == np.array(golds)
        j = (jev_m >= 0.5) == np.array(golds)
    elif q.type == "choice":
        idx = {k: i for i, k in enumerate(q.options)}
        y = np.array([idx[g] for g in golds])
        o = ours.argmax(1) == y
        j = jev_m.argmax(1) == y
    else:
        y = np.round(np.array(golds, dtype=float))
        o = ours.argmax(1) == y
        j = jev_m.argmax(1) == y
    return {
        "n": int(len(o)),
        "both_correct": float((o & j).mean()),
        "both_wrong": float((~o & ~j).mean()),
        "we_fix_jev_errors": float((o & ~j).mean()),
        "we_break_jev_correct": float((~o & j).mean()),
        "jev_error_rate": float((~j).mean()),
        "share_of_jev_errors_we_fix": float((o & ~j).sum() / max(1, (~j).sum())),
    }


# ---- evaluation helpers ----------------------------------------------------------------------------
def _train(q, tr, va, cfg, head) -> TrainResult:
    res = train_with_oom_retry(q, tr, va, cfg, qid=head, log=lambda m: rprint("  " + m))
    if res is None:
        raise RuntimeError("training failed: out of memory on every device")
    return res


def eval_on(q: Question, res: TrainResult, evalset: list[Example], head: str, groups: list[str] | None) -> tuple[np.ndarray, dict]:
    from ownyourjev.train import model_device, pick_device

    if not model_device(res.model).startswith("cuda") and torch.cuda.is_available():
        res.model.to(pick_device())
    bs = 8 if res.model.max_len > 512 else (32 if len(res.model.question.options) > 50 else 64)
    probs = predict_probs(res.model, res.tokenizer, evalset, batch_size=bs)
    golds = [e.gold[head] for e in evalset]
    return probs, evaluate(q, golds, probs, groups)


def ci_eval(q: Question, evalset: list[Example], head: str, probs: np.ndarray, groups: list[str] | None, metric: str | None = None) -> dict:
    return evaluate_with_ci(q, [e.gold[head] for e in evalset], probs, groups, iters=1000, metric=metric)


def search_cfg(task, pool_n: int, stages: str) -> SearchConfig:
    long = task.spec.long_input
    cfg = SearchConfig(max_len=1536 if long else 256, batch_size=4 if long else 16, grad_accum=4 if long else 1)
    cfg.ladder_frozen = [n for n in cfg.ladder_frozen if n < pool_n] + [pool_n]
    cfg.ladder_ft = [n for n in cfg.ladder_ft if n < pool_n] + [pool_n]
    if long:
        cfg.hpo_trials = 6
    return cfg


def secondary_head_probs(qh: Question, h: str, dh: list[Example], evalset: list[Example], folds, cfg: TrainConfig) -> tuple[np.ndarray, float]:
    """Eval-set probabilities for a secondary head; cross-fitted (never trained on the scored texts) when the
    task has no separate training pool."""
    shape = (len(evalset),) if qh.type == "noul" else (len(evalset), qh.num_outputs)
    if not folds:
        htr, hva = train_val_split(dh, 0.1, seed=0)
        res = _train(qh, htr, hva, cfg, h)
        p, _ = eval_on(qh, res, evalset, h, None)
        gpu = res.train_s
        del res
        torch.cuda.empty_cache()
        return p, gpu
    out = np.zeros(shape)
    gpu = 0.0
    for f, (tr_idx, held_idx) in enumerate(folds):
        tr_uids = {evalset[i].uid for i in tr_idx}
        ftr, fva = train_val_split([e for e in dh if e.uid in tr_uids], 0.1, seed=f)
        res = _train(qh, ftr, fva, cfg, h)
        p, _ = eval_on(qh, res, [evalset[i] for i in held_idx], h, None)
        out[held_idx] = p
        gpu += res.train_s
        del res
        torch.cuda.empty_cache()
    return out, gpu


# ---- per task --------------------------------------------------------------------------------------
def run_task(name: str, *, variants: list[str], pool_cap: int, stages: str, limit: int | None, max_cost: float | None, out: Path, skip_label: bool) -> dict:
    task = get_task(name)
    spec = task.spec
    questions = spec.parsed
    head = spec.primary_head
    q = questions[head]
    metric = spec.primary_metric
    base = BASELINES.get(name, {})
    out_dir = out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/silver") / name
    rng = random.Random(0)
    rprint(f"\n[bold cyan]== {name}[/bold cyan] ({spec.mode}, {metric}; Jev {base.get('jev')} {base.get('ci95')})")

    pool_all = task.pool()
    evalset = task.eval()
    if limit:
        evalset = stratified_subsample(evalset, limit, seed=0)
        pool_cap = min(pool_cap, limit)
    groups = [e.group for e in evalset] if spec.grouped else None
    jev = jev_answers(spec.questions, evalset, head)
    jev_m = jev_probs_matrix(q, jev, evalset) if jev else None
    jev_recomputed = evaluate(q, [e.gold[head] for e in evalset], jev_m, groups) if jev_m is not None else {}
    if jev_recomputed:
        rprint(f"  Jev recomputed on this eval set from the paper's responses: {metric}={jev_recomputed.get(metric, float('nan')):.4f}")

    # cross-fit: folds over the eval set itself
    folds = kfold(evalset, spec.cross_fit_folds, seed=0, grouped=spec.grouped) if spec.cross_fit_folds else None
    if folds:
        # label the whole set; the search trains on fold 0's training part and scores fold 0's held-out part
        pool = list(evalset)
        fold0_train_uids = {evalset[i].uid for i in folds[0][0]}
        eval_for_search = [evalset[i] for i in folds[0][1]]
    else:
        pool = stratified_subsample(pool_all, min(pool_cap, len(pool_all)), seed=0)
        eval_for_search = evalset
    # the seed examples are always held out of the pool, so every variant trains on the same texts
    # whether or not the seeded variant is run
    seeds = None
    if True:
        seeds = rng.sample(pool, min(N_SEEDS, len(pool) // 5))
        seed_uids = {e.uid for e in seeds}
        seeds = with_gold_labels(seeds)
        pool = [e for e in pool if e.uid not in seed_uids]

    result: dict[str, Any] = {
        "task": name, "mode": spec.mode, "metric": metric, "head": head, "n_eval": len(evalset), "n_pool": len(pool),
        "jev_published": base, "jev_recomputed": jev_recomputed, "variants": {}, "cross_fit_folds": spec.cross_fit_folds,
    }

    LADDER_EVAL_N = 2000  # ladder/HPO trials are scored on a fixed slice; final models on the full split

    def make_test_fn(ev: list[Example], gr: list[str] | None):
        if len(ev) > LADDER_EVAL_N:
            keep = set(random.Random(1).sample(range(len(ev)), LADDER_EVAL_N))
            ev = [e for i, e in enumerate(ev) if i in keep]
            gr = [e.group for e in ev] if gr is not None else None

        def fn(res: TrainResult) -> dict:
            _, m = eval_on(q, res, ev, head, gr)
            return m
        return fn

    selected_cfg: TrainConfig | None = None
    for variant in variants:
        vdir = out_dir / variant
        if variant == "silver":
            data = label_pool(name, questions, pool, seeds=None, path=data_dir / "silver.jsonl", max_cost=max_cost, skip=skip_label, long_input=spec.long_input)
        elif variant == "seeded":
            data = label_pool(name, questions, pool, seeds=seeds, path=data_dir / "silver_seeded.jsonl", max_cost=max_cost, skip=skip_label, long_input=spec.long_input)
        else:
            data = with_gold_labels(pool)
        data = [e for e in data if e.valid and head in e.labels]
        search_data = [e for e in data if e.uid in fold0_train_uids] if folds else data
        tr, va = train_val_split(search_data, 0.1, seed=0)
        scfg = search_cfg(task, len(tr), stages)
        gr_search = [e.group for e in eval_for_search] if spec.grouped else None
        s = Search(q, head, tr, va, scfg, vdir / "search", test_fn=make_test_fn(eval_for_search, gr_search), log=lambda m: rprint("  " + m), metric=metric)
        if variant == "silver":
            rep = s.run(stages)
            selected_cfg = TrainConfig(**rep["selected"]["config"])
        else:
            # other variants: the recipe selected on silver, on a ladder + full size (no new search)
            assert selected_cfg is not None, "run the silver variant first"
            for n in scfg.ladder_ft:
                tcfg = TrainConfig(**{**selected_cfg.to_dict(), "epochs": scfg.epochs_for(n, selected_cfg.method) or selected_cfg.epochs})
                s.run_trial("L", tcfg, min(n, len(tr)))
            rep = {"selected": asdict(s.selected_from("L")), "final_seeds": [], "n_trials": len(s.trials), "gpu_seconds_total": sum(t.train_s for t in s.trials), "learning_curves": s.learning_curves(stages=("L",)), "best_by_setting": []}
            (vdir / "search" / "report.json").write_text(json.dumps(rep, indent=1))
        sel = rep["selected"]
        sel_cfg = TrainConfig(**sel["config"])
        if task.spec.long_input and sel_cfg.batch_size * sel_cfg.max_len > 8 * 1536:
            # long inputs: same effective batch, smaller micro-batches (HPO may pick 32 x 1536 tokens)
            k = max(1, (sel_cfg.batch_size * sel_cfg.max_len) // (8 * 1536))
            sel_cfg = TrainConfig(**{**sel_cfg.to_dict(), "batch_size": max(1, sel_cfg.batch_size // k), "grad_accum": sel_cfg.grad_accum * k})
        # final predictions on the full eval set (cross-fit: retrain per fold)
        if folds:
            probs_full = np.zeros((len(evalset),) if q.type == "noul" else (len(evalset), q.num_outputs))
            fold_gpu = 0.0
            for f, (tr_idx, held_idx) in enumerate(folds):
                fold_data = [e for e in data if e.uid in {evalset[i].uid for i in tr_idx}]
                ftr, fva = train_val_split(fold_data, 0.1, seed=f)
                res = _train(q, ftr, fva, sel_cfg, head)
                fold_gpu += res.train_s
                held = [evalset[i] for i in held_idx]
                p, _ = eval_on(q, res, held, head, None)
                probs_full[held_idx] = p
                del res
                torch.cuda.empty_cache()
            final_metrics = ci_eval(q, evalset, head, probs_full, groups, metric)
            final_metrics["gpu_s_final"] = fold_gpu
        else:
            best = s.best[2] if s.best and s.best[2] is not None and asdict(s.best[1])["key"] == sel["key"] else None
            d = s.best_model_on_disk(sel["key"])
            if best is None and d is not None:  # resumed run: reuse the selected trial's model
                from transformers import AutoTokenizer

                from ownyourjev.models.decision import DecisionModel
                from ownyourjev.train import pick_device

                m = DecisionModel.load(d, device=pick_device(2.0))
                best = TrainResult(m, AutoTokenizer.from_pretrained(d / "backbone"), sel_cfg, sel["n_train"], len(va), sel["val_loss"], sel["val_metric"], sel["val_metrics"], [], 0.0, sel["train_s"], sel["best_epoch"], sel["temperature"])
            if best is None:
                best = _train(q, tr, va, sel_cfg, head)
            probs_full, _ = eval_on(q, best, evalset, head, groups)
            final_metrics = ci_eval(q, evalset, head, probs_full, groups, metric)
            final_metrics["gpu_s_final"] = best.train_s
            if variant == "silver":
                best.model.save(out_dir / "model", best.tokenizer)
            del best
            torch.cuda.empty_cache()
        np.save(vdir / "eval_probs.npy", probs_full)
        other_heads = {}
        if len(questions) > 1 and variant == "silver":
            for h, qh in questions.items():
                if h == head:
                    continue
                dh = [e for e in data if h in e.labels]
                p, gpu = secondary_head_probs(qh, h, dh, evalset, folds, sel_cfg)
                m = evaluate(qh, [e.gold[h] for e in evalset], p, groups)
                other_heads[h] = {**m, "gpu_s": gpu, "cross_fit": bool(folds)}
                np.save(vdir / f"eval_probs_{h}.npy", p)
        result["variants"][variant] = {
            "n_train": len(tr), "n_val": len(va), "recipe": sel["config"], "selected_trial": sel,
            "final": final_metrics, "other_heads": other_heads,
            "final_seeds": [{"seed": t["seed"], "val": t["val_metric"], "test": (t.get("test_metrics") or {}).get(metric)} for t in rep["final_seeds"]],
            "gpu_seconds_search": rep["gpu_seconds_total"], "n_trials": rep["n_trials"],
            "learning_curves": rep["learning_curves"], "best_by_setting": rep["best_by_setting"],
            "overlap_with_jev": overlap(q, [e.gold[head] for e in evalset], probs_full, jev_m),
        }
        v = final_metrics["primary_value"]
        rprint(f"  [bold]{variant}[/bold]: {metric}={v:.4f} CI {final_metrics['ci95']}  vs Jev {base.get('jev')}  recipe={sel['config']['backbone']}/{sel['config']['method']} n={sel['n_train']}")
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    b = Budget()
    result["spend_usd_total_so_far"] = b.spent()
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def run_benchmark(tasks: str, *, variants: list[str], pool_cap: int, stages: str, limit: int | None, max_cost: float | None, out: Path, skip_label: bool) -> None:
    for name in resolve_tasks(tasks):
        try:
            run_task(name, variants=variants, pool_cap=pool_cap, stages=stages, limit=limit, max_cost=max_cost, out=out, skip_label=skip_label)
        except RuntimeError as e:
            rprint(f"[red]{name}: {e}[/red]")
        rprint(f"  spend so far: ${Budget().spent():.4f}")
