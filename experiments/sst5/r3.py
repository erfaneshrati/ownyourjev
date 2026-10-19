import asyncio, sys
sys.path.insert(0, "experiments/sst5")
from lab import *
from calib import cv_rep
# args: tag model temperature seeds(comma or 'none')
tag, model, temp, seeds = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
g = open(f"experiments/sst5/guidance_{tag}.txt").read()
ss = [None] if seeds == "none" else [int(x) for x in seeds.split(",")]
names = [f"dev_{tag}_{model.split('/')[1]}" + ("" if s is None else f"_t{temp}_s{s}") for s in ss]
async def main():
    await asyncio.gather(*[run(n, model, DEV, guidance=g, temperature=temp, seed=s) for n, s in zip(names, ss)])
asyncio.run(main())
uids = [e.uid for e in DEV]
for n in names: report(n, verbose=False) and print(n, report(n, verbose=True))
