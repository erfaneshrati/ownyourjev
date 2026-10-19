import asyncio, sys
sys.path.insert(0, "experiments/sst5")
from lab import *
tag, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
g = open(f"experiments/sst5/guidance_{tag}.txt").read()
excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
pool = [e for e in P.pool(T) if e.uid not in excl][:n]
asyncio.run(run(f"pool_{tag}_{model.split('/')[1]}_{n}", model, pool, guidance=g, pricey=False))
