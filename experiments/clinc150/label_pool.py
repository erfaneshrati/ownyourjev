import asyncio
from benchmarks.v2 import protocol as P
from ownyourjev.data import write_jsonl
from experiments.clinc150 import blabel
from collections import Counter
pool = P.pool("clinc150")
out = asyncio.run(blabel.label(pool, seeds=P.seeds("clinc150"), purpose="clinc-pool-v1", concurrency=24))
write_jsonl("data/v2/clinc150/pool_labeled.jsonl", out)
c = Counter((o.labels or {}).get("answer") for o in out)
print("n", len(out), "oos", c["oos"], "none", c[None], "min/max per class", min(c.values()), max(c.values()))
