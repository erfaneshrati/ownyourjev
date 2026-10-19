"""Label pool examples with the decimal qwen labeler (no seeds). Usage: label_pool.py <out_name> <which> [n]
which: silver (the v1 silver uids) | rest (pool uids not in silver, first n in a fixed shuffle)."""
import asyncio, random, sys
from common import *
from label_dec import label
from ownyourjev.llm import OpenRouterClient
from fastbudget import FastBudget, FastCache
name, which = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
hold = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
pool = [e for e in protocol.pool(T) if e.uid not in hold]
sil = {e.uid for e in load_jsonl(LD / "silver_v1.jsonl")}
if which == "silver":
    exs = [e for e in pool if e.uid in sil]
else:
    exs = [e for e in pool if e.uid not in sil]
    random.Random(0).shuffle(exs)
if n: exs = exs[:n]
print(len(exs), flush=True)
async def main():
    c = OpenRouterClient(model="qwen/qwen3.7-flash", concurrency=24, timeout=60, budget=FastBudget(), cache=FastCache())
    out = await label(c, [Example(e.uid, e.state, None, None, e.group, {}) for e in exs], f"{T}:pool_dec_qwen")
    await c.aclose(); c.budget.flush(); c.cache.flush()
    save_jsonl(LD / f"{name}.jsonl", [e for e in out if e.labels])
    print("failed", sum(e.labels is None for e in out), "cost", c.stats["cost"], flush=True)
asyncio.run(main())
