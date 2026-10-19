import asyncio, sys
sys.path.insert(0, "experiments/sst5")
from lab import *
models = sys.argv[1].split(",")
async def main():
    await asyncio.gather(*[run(f"dev_base_{m.split('/')[1]}", m, DEV, pricey=True) for m in models])
asyncio.run(main())
for m in models: report(f"dev_base_{m.split('/')[1]}")
