"""Ledger/cache wrappers for large labeling jobs on the shared SQLite files.

The library commits one transaction per call to cache/budget.db and cache/llm_cache.db from inside the asyncio
event loop; with many agents writing the same files, each commit waits on the database lock and the whole
job crawls (~1 call/s). These wrappers keep the same caps and the same rows, but:
  - check() reuses ledger totals for `ttl` seconds plus spend recorded locally since then,
  - record()/put() are buffered and written by a background thread in one transaction every `every` seconds.
Call flush() before exit (also registered with atexit)."""
import atexit, json, threading, time
from ownyourjev.llm.budget import Budget, BudgetExceeded
from ownyourjev.llm.client import ResponseCache


class _Flusher:
    def _start(self, every):
        self._buf, self._blk, self._every = [], threading.Lock(), every
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start()
        atexit.register(self.flush)

    def _loop(self):
        while not self._stop:
            time.sleep(self._every); self.flush()

    def flush(self):
        with self._blk:
            rows, self._buf = self._buf, []
        if rows:
            with self._lock:
                self._conn.executemany(self._sql, rows); self._conn.commit()


class FastBudget(_Flusher, Budget):
    _sql = "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)"

    def __init__(self, *a, ttl=5.0, every=2.0, **k):
        Budget.__init__(self, *a, **k); self.ttl = ttl; self._t = 0.0; self._s = self._ss = 0.0; self._local = 0.0
        self._start(every)

    def check(self, estimated_usd: float = 0.0) -> None:
        if time.time() - self._t > self.ttl:
            self.flush()
            self._s, self._ss, self._t = Budget.spent(self), Budget.scope_spent(self), time.time()
        if self._s + estimated_usd > self.cap:
            raise BudgetExceeded(f"budget cap {self.cap:.2f} exceeded: spent {self._s:.4f}")
        if self.scope and self.scope_cap is not None and self._ss + estimated_usd > self.scope_cap:
            raise BudgetExceeded(f"scope {self.scope!r} cap {self.scope_cap:.2f} exceeded: spent {self._ss:.4f}")

    def record(self, *, purpose, model, prompt_tokens, completion_tokens, cost_usd, cached=False, request_hash=""):
        if self.scope and not purpose.startswith(f"{self.scope}:"):
            purpose = f"{self.scope}:{purpose}"
        with self._blk:
            self._buf.append((time.time(), purpose, model, prompt_tokens, completion_tokens, cost_usd, int(cached), request_hash))
        if not cached:
            self._s += cost_usd; self._ss += cost_usd


class FastCache(_Flusher, ResponseCache):
    _sql = "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)"

    def __init__(self, *a, every=2.0, **k):
        ResponseCache.__init__(self, *a, **k); self._pending = {}; self._start(every)

    def get(self, key):
        if key in self._pending:
            return self._pending[key]
        return ResponseCache.get(self, key)

    def put(self, key, model, content, usage):
        self._pending[key] = (content, usage)
        with self._blk:
            self._buf.append((key, model, content, json.dumps(usage), time.time()))
