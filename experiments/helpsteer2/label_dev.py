"""Compare labeler variants on the 200 dev examples (per-dim Spearman; dev halves A/B and all)."""
import asyncio, json, sys
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient

GUIDE = (Path(__file__).parent / "guidance.txt").read_text() if (Path(__file__).parent / "guidance.txt").exists() else None
VARIANTS = {
    "flash_hard": dict(model="deepseek/deepseek-v4.1-flash", dist=False, seeds=False, guide=False),
    "flash_soft": dict(model="deepseek/deepseek-v4.1-flash", dist=True, seeds=False, guide=False),
    "flash_soft_seeds": dict(model="deepseek/deepseek-v4.1-flash", dist=True, seeds=True, guide=False),
    "flash_soft_seeds_guide": dict(model="deepseek/deepseek-v4.1-flash", dist=True, seeds=True, guide=True),
    "flash_hard_seeds_guide": dict(model="deepseek/deepseek-v4.1-flash", dist=False, seeds=True, guide=True),
    "qwen_soft_seeds_guide": dict(model="qwen/qwen3.7-flash", dist=True, seeds=True, guide=True),
    "pro_soft_seeds_guide": dict(model="deepseek/deepseek-v4-pro", dist=True, seeds=True, guide=True),
    "pro_hard_seeds_guide": dict(model="deepseek/deepseek-v4-pro", dist=False, seeds=True, guide=True),
    "haiku_soft_seeds_guide": dict(model="anthropic/claude-haiku-5.5", dist=True, seeds=True, guide=True),
    "gemlite_soft_seeds_guide": dict(model="google/gemini-2.5-flash-lite", dist=True, seeds=True, guide=True),
    "luna_hard_seeds_guide": dict(model="openai/gpt-5.6-luna", dist=False, seeds=True, guide=True),
}

async def run(name):
    v = VARIANTS[name]
    c = OpenRouterClient(model=v["model"], concurrency=16)
    dev = protocol.dev(T)
    inp = [Example(e.uid, e.state, None, None, e.group, {}) for e in dev]
    out = await label_examples(c, Q, inp, purpose=f"{T}:dev_{name}", seeds=protocol.seeds(T) if v["seeds"] else None,
                               guidance=GUIDE if v["guide"] else None, distribution=v["dist"], progress=False)
    await c.aclose()
    nfail = sum(e.labels is None for e in out)
    save_jsonl(LD / f"dev_{name}.jsonl", out)
    report(f"{name} (fail {nfail}, ${c.stats['cost']:.4f})", out, dev)

for n in sys.argv[1:]:
    asyncio.run(run(n))
