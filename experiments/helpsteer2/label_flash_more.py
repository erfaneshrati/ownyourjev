"""More deepseek-flash hard labels (same prompt as v1 silver: label_examples defaults, no seeds) on pool texts the v1
silver did not cover, routed to the cheapest provider. Usage: label_flash_more.py <out_name> <n> [offset]"""
import asyncio, random, sys
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
from fastbudget import FastBudget, FastCache
name, n = sys.argv[1], int(sys.argv[2]); off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sil = {e.uid for e in load_jsonl(LD / "silver_v1.jsonl")}
exs = [e for e in protocol.pool(T) if e.uid not in sil]
random.Random(0).shuffle(exs); exs = exs[off:off + n]
async def main():
    c = OpenRouterClient(model="deepseek/deepseek-v4.1-flash", concurrency=24, timeout=60, budget=FastBudget(), cache=FastCache(),
                         extra_body={"provider": {"sort": "price"}})
    out = await label_examples(c, Q, [Example(e.uid, e.state, None, None, e.group, {}) for e in exs], purpose=f"{T}:pool_flash_more", progress=False)
    await c.aclose(); c.budget.flush(); c.cache.flush()
    save_jsonl(LD / f"{name}.jsonl", [e for e in out if e.labels])
    print("n", len(out), "failed", sum(e.labels is None for e in out), "cost", c.stats["cost"], "per call", c.stats["cost"] / max(1, c.stats["calls"]), flush=True)
asyncio.run(main())
