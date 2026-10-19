"""Label the whole unlabeled pool with the dev-chosen prompt (flash_g1: deepseek-v4.1-flash, guidance v1, soft).
Sharded across processes (usage: label_pool.py SHARD NSHARDS; then label_pool.py merge NSHARDS): the client does
synchronous sqlite work (cache + ledger) inside the event loop, so one process tops out at ~60 calls/min."""
import asyncio, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import GUIDANCE_V1, LAB, QS, TASK, save, load
from benchmarks.v2 import protocol as P
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
from fast_budget import FastBudget


async def main(shard, n):
    c = OpenRouterClient(model="deepseek/deepseek-v4.1-flash", concurrency=32, budget=FastBudget())
    pool = P.pool(TASK)[shard::n]; out = []
    for i in range(0, len(pool), 512):
        out += await label_examples(c, QS, pool[i:i + 512], purpose=f"{TASK}:pool:flash_g1", guidance=GUIDANCE_V1, distribution=True, progress=False)
        print(shard, i + 512, round(c.stats["cost"], 4), flush=True)
    await c.aclose()
    save(out, LAB / f"pool_flash_g1.shard{shard}.jsonl")
    print("done", c.stats, "fail", sum(e.labels is None for e in out))

if sys.argv[1] == "merge":
    import json
    n = int(sys.argv[2]); recs = {}
    for s in range(n):
        recs.update(load(LAB / f"pool_flash_g1.shard{s}.jsonl"))
    with (LAB / "pool_flash_g1.jsonl").open("w") as f:
        for e in P.pool(TASK):
            f.write(json.dumps(recs[e.uid]) + "\n")
    print("merged", len(recs), "fail", sum(r["labels"] is None for r in recs.values()))
else:
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
