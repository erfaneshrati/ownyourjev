"""Label the 200 dev examples with labeling variants and report agreement with dev gold.
Usage: uv run python experiments/toxigen/label_dev.py VARIANT [VARIANT ...]"""
from __future__ import annotations

import asyncio
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import GUIDANCE_V1, LAB, QS, RES, TASK, dev_report, save, seeds, soft_of, load  # noqa: E402
from benchmarks.v2 import protocol as P  # noqa: E402
from ownyourjev.label import label_examples  # noqa: E402
from ownyourjev.llm import OpenRouterClient  # noqa: E402

FLASH = "deepseek/deepseek-v4.1-flash"
VARIANTS = {
    # name: (model, seeds: None|'soft'|'hard', guidance, distribution, extra_body)
    "flash_plain": (FLASH, None, None, True, None),
    "flash_seeds": (FLASH, "soft", None, True, None),
    "flash_seeds_g1": (FLASH, "soft", "g1", True, None),
    "flash_g1": (FLASH, None, "g1", True, None),
    "flash_hseeds_g1_hard": (FLASH, "hard", "g1", False, None),
    "qwen_seeds_g1": ("qwen/qwen3.7-flash", "soft", "g1", True, None),
    "gemlite_seeds_g1": ("google/gemini-2.5-flash-lite", "soft", "g1", True, None),
    "haiku_seeds_g1": ("anthropic/claude-haiku-5.5", "soft", "g1", True, None),
    "pro_seeds_g1": ("deepseek/deepseek-v4-pro", "soft", "g1", True, None),
    "luna_seeds_g1": ("openai/gpt-5.6-luna", "soft", "g1", True, None),
    "gem31_seeds_g1": ("google/gemini-3.1-flash-lite", "soft", "g1", True, None),
}
GUIDE = {"g1": GUIDANCE_V1}


async def run(name: str):
    model, sd, g, dist, eb = VARIANTS[name]
    client = OpenRouterClient(model=model, extra_body=eb)
    dev = P.dev(TASK)
    exs = await label_examples(client, QS, dev, purpose=f"{TASK}:dev:{name}", seeds=seeds(sd == "soft") if sd else None,
                               guidance=GUIDE.get(g), distribution=dist, progress=False)
    await client.aclose()
    save(exs, LAB / f"dev_{name}.jsonl")
    recs = load(LAB / f"dev_{name}.jsonl")
    ok = [i for i, e in enumerate(dev) if recs[e.uid]["labels"] is not None]
    sp = [soft_of(recs[dev[i].uid]) for i in ok]
    p = np.array([s[0] for s in sp]); d = np.stack([s[1] for s in sp])
    dv = [dev[i] for i in ok]
    rep = {"variant": name, "n_fail": len(dev) - len(ok), "cost": client.stats["cost"], "all": dev_report(p, d, dv),
           "A": dev_report(p, d, dv, np.array([i % 2 == 0 for i in range(len(dv))])),
           "B": dev_report(p, d, dv, np.array([i % 2 == 1 for i in range(len(dv))]))}
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in rep.items()}))
    with (RES / "dev_labeler_log.jsonl").open("a") as f:
        f.write(json.dumps(rep) + "\n")


async def main():
    await asyncio.gather(*(run(n) for n in sys.argv[1:]))

asyncio.run(main())
