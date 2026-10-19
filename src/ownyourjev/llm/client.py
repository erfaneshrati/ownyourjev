"""OpenRouter chat client with JSON output, a request cache and the spend ledger.

Design:
- Every request is cached on disk by a hash of (model, messages, response format). Re-running a labeling job
  is free.
- Before sending, the cost is estimated from the model's price list (fetched once from the public
  ``/api/v1/models`` endpoint) and checked against the budget cap. After the response the exact
  ``usage.cost`` reported by OpenRouter is recorded.
- ``FakeLLMClient`` has the same interface for offline tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import httpx
from dotenv import find_dotenv, load_dotenv

from ownyourjev.llm.budget import Budget
from ownyourjev.spec import canonical_json

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
PRICE_CACHE_TTL = 24 * 3600


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(t[start:])
    return obj


class LLMClient(Protocol):
    model: str

    async def complete_json(self, messages: list[dict], *, purpose: str, max_tokens: int = 512) -> Any: ...


class ResponseCache:
    def __init__(self, path: str | Path | None = None):
        path = path or os.environ.get("OYJ_LLM_CACHE") or (Path(__file__).resolve().parents[3] / "cache" / "llm_cache.db")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, content TEXT, usage TEXT, ts REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> tuple[str, dict] | None:
        with self._lock:
            row = self._conn.execute("SELECT content, usage FROM responses WHERE key = ?", (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put(self, key: str, model: str, content: str, usage: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)", (key, model, content, json.dumps(usage), time.time())
            )
            self._conn.commit()


class OpenRouterClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        budget: Budget | None = None,
        cache: ResponseCache | None = None,
        concurrency: int = 16,
        max_retries: int = 4,
        temperature: float = 0.0,
        extra_body: dict | None = None,
        timeout: float = 120.0,
    ):
        load_dotenv(find_dotenv(usecwd=True))
        self.model = model or os.environ.get("OYJ_LABELER_MODEL", "deepseek/deepseek-v4.1-flash")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.budget = budget or Budget()
        self.cache = cache or ResponseCache()
        self.sem: asyncio.Semaphore | None = None
        self.max_retries = max_retries
        self.temperature = temperature
        # reasoning models burn output tokens on hidden thinking; labels are short, so disable it by default
        # route to the fastest provider (the default cheapest route was ~15x slower in our tests); cap price
        # so a fast-but-expensive outlier can never be chosen
        self.extra_body = {
            "reasoning": {"enabled": False},
            "provider": {"sort": "throughput", "max_price": {"prompt": 0.2, "completion": 0.8}},
            **(extra_body or {}),
        }
        self.timeout = timeout
        self._prices: tuple[float, float] | None = None
        self._concurrency = concurrency
        self._http: httpx.AsyncClient | None = None
        self._loop = None
        self.stats = {"calls": 0, "cached": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    def _ensure_loop_state(self) -> None:
        """httpx clients and semaphores are bound to one event loop; recreate them when the client is reused
        across several asyncio.run() calls (previously this silently turned every call into a failure)."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self.sem = asyncio.Semaphore(self._concurrency)
            self._http = httpx.AsyncClient(
                base_url=OPENROUTER_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/erfaneshrati/ownyourjev",
                    "X-Title": "ownyourjev",
                },
                timeout=self.timeout,
            )

    # ---- pricing -------------------------------------------------------------------------------------
    def prices(self) -> tuple[float, float]:
        """(usd per prompt token, usd per completion token) for self.model."""
        if self._prices is not None:
            return self._prices
        p = Path(__file__).resolve().parents[3] / "cache" / "openrouter_prices.json"
        data: dict = {}
        if p.is_file() and time.time() - p.stat().st_mtime < PRICE_CACHE_TTL:
            data = json.loads(p.read_text())
        if self.model not in data:
            r = httpx.get(f"{OPENROUTER_BASE}/models", timeout=30)
            r.raise_for_status()
            data = {
                m["id"]: (float(m["pricing"].get("prompt", 0)), float(m["pricing"].get("completion", 0)))
                for m in r.json()["data"]
            }
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
        if self.model not in data:
            raise RuntimeError(f"model {self.model} not found on OpenRouter")
        self._prices = tuple(data[self.model])  # type: ignore[assignment]
        return self._prices

    def estimate_cost(self, messages: list[dict], max_tokens: int) -> float:
        pin, pout = self.prices()
        n_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = n_chars / 3.5 + 20  # conservative
        return prompt_tokens * pin + max_tokens * pout

    # ---- calls ----------------------------------------------------------------------------------------
    async def complete_json(self, messages: list[dict], *, purpose: str, max_tokens: int = 512) -> Any:
        text, _ = await self.complete(messages, purpose=purpose, max_tokens=max_tokens, json_mode=True)
        return _extract_json(text)

    async def complete(
        self, messages: list[dict], *, purpose: str, max_tokens: int = 512, json_mode: bool = True
    ) -> tuple[str, dict]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
            **self.extra_body,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        extra = {k: v for k, v in self.extra_body.items() if k != "provider" and not (k == "reasoning" and v == {"enabled": False})}
        parts = [self.model, canonical_json(messages), canonical_json(body.get("response_format")), str(self.temperature)]
        if extra:  # e.g. a sampling seed; empty for default calls so older cache entries keep matching
            parts.append(canonical_json(extra))
        if max_tokens != 512:  # a reply truncated by a small token limit must not be reused at a larger limit
            parts.append(f"max_tokens={max_tokens}")
        key = _hash(*parts)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cached"] += 1
            return hit
        est = self.estimate_cost(messages, max_tokens)
        self._ensure_loop_state()
        async with self.sem:
            self.budget.check(est)  # raises BudgetExceeded; includes requests already in flight
            self.budget.reserve(est)
            try:
                return await self._send(body, key, purpose)
            finally:
                self.budget.release(est)

    async def _send(self, body: dict, key: str, purpose: str) -> tuple[str, dict]:
        if True:
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    r = await self._http.post("/chat/completions", json=body)
                    if r.status_code in (429, 500, 502, 503, 504, 408):
                        raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:
                        raise RuntimeError(f"OpenRouter error: {data['error']}")
                    content = data["choices"][0]["message"].get("content") or ""
                    usage = data.get("usage") or {}
                    cost = float(usage.get("cost") or 0.0)
                    if not usage.get("cost"):
                        pin, pout = self.prices()
                        cost = usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout
                    self.budget.record(
                        purpose=purpose,
                        model=self.model,
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        cost_usd=cost,
                        request_hash=key,
                    )
                    self.stats["calls"] += 1
                    self.stats["cost"] += cost
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
                    if content.strip():
                        self.cache.put(key, self.model, content, usage)
                    return content, usage
                except (httpx.HTTPError, RuntimeError, KeyError) as e:
                    last_err = e
                    await asyncio.sleep(min(30, 2**attempt + 0.5))
            raise RuntimeError(f"OpenRouter call failed after {self.max_retries} attempts: {last_err}")

    async def map(
        self,
        items: list[Any],
        fn: Callable[[Any], Awaitable[Any]],
        *,
        desc: str = "",
        progress: bool = True,
    ) -> list[Any]:
        """Run ``fn`` over items concurrently (bounded by the semaphore inside ``complete``)."""
        from tqdm.asyncio import tqdm_asyncio

        coros = [fn(it) for it in items]
        if progress:
            return await tqdm_asyncio.gather(*coros, desc=desc)
        return await asyncio.gather(*coros)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http, self._loop = None, None


class FakeLLMClient:
    """Offline stand-in: answers with a user-supplied function of the messages."""

    def __init__(self, responder: Callable[[list[dict]], Any], model: str = "fake"):
        self.responder = responder
        self.model = model
        self.calls: list[list[dict]] = []
        self.stats = {"calls": 0, "cached": 0, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def complete_json(self, messages: list[dict], *, purpose: str, max_tokens: int = 512) -> Any:
        self.calls.append(messages)
        self.stats["calls"] += 1
        out = self.responder(messages)
        return _extract_json(out) if isinstance(out, str) else out

    async def map(self, items, fn, *, desc: str = "", progress: bool = False):
        return await asyncio.gather(*[fn(it) for it in items])

    async def aclose(self) -> None:
        return None
