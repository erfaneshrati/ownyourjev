import asyncio, sys
sys.path.insert(0, "experiments/sst5")
from lab import *
from calib import cv_rep
tag = sys.argv[1]; g = open(f"experiments/sst5/guidance_{tag}.txt").read(); models = sys.argv[2].split(",")
async def main():
    await asyncio.gather(*[run(f"dev_{tag}_{m.split('/')[1]}", m, DEV, guidance=g, pricey=True) for m in models])
asyncio.run(main())
uids=[e.uid for e in DEV]
for m in models:
    n=f"dev_{tag}_{m.split('/')[1]}"; report(n); print("cv-bias", round(cv_rep(probs_of(load(n),uids),DEVY),3))
