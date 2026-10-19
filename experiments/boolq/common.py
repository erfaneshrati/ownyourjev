"""Shared helpers for the boolq v2 experiment."""
from __future__ import annotations
import asyncio, json, os
from pathlib import Path
import numpy as np
from benchmarks.v2 import protocol
from ownyourjev.data import Example, read_jsonl, write_jsonl

TASK = "boolq"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/v2/boolq"
RES = ROOT / "benchmarks/results_v2/boolq"
Q = protocol.task_spec(TASK).parsed
QID = "answer"


def gold_uids() -> set[str]:
    return {e.uid for e in protocol.seeds(TASK)} | {e.uid for e in protocol.dev(TASK)}


def dev_halves():
    d = protocol.dev(TASK)
    return d[:100], d[100:]


def acc(golds, labels) -> float:
    return float(np.mean([g == l for g, l in zip(golds, labels)]))


def client(model: str, **kw):
    from ownyourjev.llm import OpenRouterClient
    eb = kw.pop("extra_body", {})
    if model not in ("deepseek/deepseek-v4.1-flash", "qwen/qwen3.7-flash", "google/gemini-2.5-flash-lite", "anthropic/claude-haiku-5.5"):
        eb = {"provider": {"sort": "throughput"}, **eb}  # default max_price filter would exclude pricier models
    return OpenRouterClient(model=model, extra_body=eb, **kw)
