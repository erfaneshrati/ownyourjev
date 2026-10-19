"""Spend ledger with a hard cap.

Every paid call is recorded in SQLite with the cost reported by OpenRouter (``usage.cost``). Before a call is
sent, ``check()`` must pass with a conservative pre-flight estimate; once the recorded spend reaches the cap
nothing else is sent. The cap defaults to ``OYJ_BUDGET_USD`` (15 USD).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
DEFAULT_CAP = float(os.environ.get("OYJ_BUDGET_USD", "15"))


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class SpendRow:
    purpose: str
    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class Budget:
    def __init__(self, db_path: str | Path | None = None, cap_usd: float | None = None, scope: str | None = None, scope_cap_usd: float | None = None):
        # one ledger for the whole repo, whatever the working directory
        db_path = db_path or os.environ.get("OYJ_BUDGET_DB") or (Path(__file__).resolve().parents[3] / "cache" / "budget.db")
        self.db_path = Path(db_path)
        # an optional scope (e.g. one experiment) with its own cap, on top of the global cap
        self.scope = scope if scope is not None else os.environ.get("OYJ_BUDGET_SCOPE") or None
        sc = scope_cap_usd if scope_cap_usd is not None else os.environ.get("OYJ_SCOPE_CAP_USD")
        self.scope_cap = float(sc) if sc not in (None, "") else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = DEFAULT_CAP if cap_usd is None else float(cap_usd)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                ts REAL, purpose TEXT, model TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
                cost_usd REAL, cached INTEGER, request_hash TEXT)"""
        )
        self._conn.commit()

    # ---- accounting -------------------------------------------------------------------------------
    REFRESH_S = 5.0  # other processes' spend is picked up at most this stale; our own spend is counted at once

    def _refresh(self) -> None:
        now = time.time()
        if now - getattr(self, "_t", 0.0) < self.REFRESH_S:
            return
        with self._lock:
            tot = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE cached = 0").fetchone()[0]
            sc = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE cached = 0 AND purpose LIKE ?", (f"{self.scope}:%",)
            ).fetchone()[0] if self.scope else 0.0
        self._tot, self._sc, self._t = float(tot), float(sc), now

    def spent(self) -> float:
        self._refresh()
        return self._tot

    def remaining(self) -> float:
        return self.cap - self.spent()

    def scope_spent(self) -> float:
        if not self.scope:
            return 0.0
        self._refresh()
        return self._sc

    # requests sent but not yet recorded (per process): counted against the caps so that many concurrent
    # requests cannot all pass the check before any of them is recorded (that leaked a cap by ~$0.13)
    def reserve(self, usd: float) -> None:
        with self._lock:
            self._pending = getattr(self, "_pending", 0.0) + usd

    def release(self, usd: float) -> None:
        with self._lock:
            self._pending = max(0.0, getattr(self, "_pending", 0.0) - usd)

    def check(self, estimated_usd: float = 0.0) -> None:
        estimated_usd += getattr(self, "_pending", 0.0)
        spent = self.spent()
        if spent + estimated_usd > self.cap:
            raise BudgetExceeded(
                f"budget cap {self.cap:.2f} USD would be exceeded: spent {spent:.4f} + estimate {estimated_usd:.4f}"
            )
        if self.scope and self.scope_cap is not None:
            ss = self.scope_spent()
            if ss + estimated_usd > self.scope_cap:
                raise BudgetExceeded(
                    f"scope {self.scope!r} cap {self.scope_cap:.2f} USD would be exceeded: spent {ss:.4f} + estimate {estimated_usd:.4f}"
                )

    def record(
        self,
        *,
        purpose: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cached: bool = False,
        request_hash: str = "",
    ) -> None:
        if self.scope and not purpose.startswith(f"{self.scope}:"):
            purpose = f"{self.scope}:{purpose}"
        if not cached:  # count our own spend immediately, without waiting for the next refresh
            self._refresh()
            self._tot += cost_usd
            if self.scope:
                self._sc += cost_usd
        with self._lock:
            self._conn.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), purpose, model, prompt_tokens, completion_tokens, cost_usd, int(cached), request_hash),
            )
            self._conn.commit()

    def summary(self) -> list[SpendRow]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT purpose, model, COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),
                          COALESCE(SUM(cost_usd),0) FROM calls WHERE cached = 0 GROUP BY purpose, model ORDER BY 6 DESC"""
            ).fetchall()
        return [SpendRow(*r) for r in rows]

    def close(self) -> None:
        self._conn.close()
