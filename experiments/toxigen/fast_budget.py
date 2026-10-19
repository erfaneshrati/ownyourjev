"""Budget wrapper that keeps the cap checks but refreshes the ledger sums at most every `ttl` seconds.
The stock Budget.check runs two full-table SUM queries (~150 ms on the shared ledger) synchronously inside the
asyncio loop for every request, which serialises a large labeling run (~50 calls/min). Between refreshes the
locally recorded spend is added, so the cap is still enforced (lag <= ttl seconds of other agents' spend)."""
import time
from ownyourjev.llm.budget import Budget, BudgetExceeded


class FastBudget(Budget):
    def __init__(self, *a, ttl: float = 10.0, **k):
        super().__init__(*a, **k)
        self.ttl = ttl; self._t = 0.0; self._g = 0.0; self._s = 0.0; self._local = 0.0

    def check(self, estimated_usd: float = 0.0) -> None:
        if time.time() - self._t > self.ttl:
            self._g, self._s, self._local, self._t = self.spent(), self.scope_spent(), 0.0, time.time()
        if self._g + self._local + estimated_usd > self.cap:
            raise BudgetExceeded(f"global cap {self.cap} would be exceeded")
        if self.scope and self.scope_cap is not None and self._s + self._local + estimated_usd > self.scope_cap:
            raise BudgetExceeded(f"scope {self.scope} cap {self.scope_cap} would be exceeded")

    def record(self, **kw) -> None:
        super().record(**kw)
        self._local += float(kw.get("cost_usd", 0.0))
