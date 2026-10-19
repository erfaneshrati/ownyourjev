"""Compact 'decimal rating' labeler: one fractional score 0-4 per attribute (cheap output, continuous signal).
A rating x is stored as hard label round(x) and soft target split between floor(x) and ceil(x)."""
import asyncio, json, math, sys
from common import *
from ownyourjev.label import SYSTEM_PREAMBLE, seeds_block
from ownyourjev.llm import OpenRouterClient
from ownyourjev.spec import render_state

FMT = ("Return exactly this JSON object: {" + ", ".join(f'"{a}": <rating>' for a in A) + "}. Each rating is a number from 0 to 4 "
       "with one decimal: the level index you expect human annotators to give. Use the decimal to express uncertainty between adjacent levels "
       "(e.g. 3.7 = most likely 4, possibly 3; 2.5 = equally 2 or 3).")

def system(seeds=None, guidance=None):
    parts = [SYSTEM_PREAMBLE, "", "## Questions"] + [f"### {a}\n{Q[a].describe()}\n" for a in A]
    if guidance: parts.append("## Labeling guidelines\n" + guidance.strip() + "\n")
    parts.append("## Output format\n" + FMT)
    if seeds: parts += ["", seeds_block(Q, seeds).replace("ANSWER:", "ANSWER (integer gold levels):")]
    return "\n".join(parts)

def soft_of(x):
    x = min(4.0, max(0.0, float(x))); lo = int(math.floor(x)); hi = min(4, lo + 1); f = x - lo
    s = [0.0] * 5; s[lo] += 1 - f; s[hi] += f
    return s

async def label(client, exs, purpose, seeds=None, guidance=None, max_chars=None):
    sysmsg = system(seeds, guidance)
    async def one(e):
        st = e.state
        if max_chars:
            st = {"prompt": st["prompt"][-max_chars:], "response": st["response"][:max_chars]}
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "## STATE\n" + render_state(st) + "\n\n## ANSWER (JSON only)"}]
        for _ in range(2):
            try:
                raw = await client.complete_json(msgs, purpose=purpose, max_tokens=80)
                vals = {a: float(raw[a]) for a in A}
                return Example(e.uid, e.state, {a: int(round(min(4, max(0, v)))) for a, v in vals.items()}, None, e.group,
                               {"labeler": client.model, "dec": vals, "soft": {a: soft_of(v) for a, v in vals.items()}})
            except Exception as ex:
                err = str(ex)
        return Example(e.uid, e.state, None, None, e.group, {"label_error": err[:200]})
    return await client.map(exs, one, desc=purpose, progress=False)

VARIANTS = {
    "dec_flash": dict(model="deepseek/deepseek-v4.1-flash", seeds=False, guide=False),
    "dec_flash_seeds": dict(model="deepseek/deepseek-v4.1-flash", seeds=True, guide=False),
    "dec_qwen": dict(model="qwen/qwen3.7-flash", seeds=False, guide=False),
    "dec_qwen_seeds": dict(model="qwen/qwen3.7-flash", seeds=True, guide=False),
    "dec_pro": dict(model="deepseek/deepseek-v4-pro", seeds=False, guide=False),
    "dec_luna": dict(model="openai/gpt-5.6-luna", seeds=False, guide=False),
    "dec_haiku": dict(model="anthropic/claude-haiku-5.5", seeds=False, guide=False),
}

async def run(name, seed=None):
    v = VARIANTS[name]
    c = OpenRouterClient(model=v["model"], concurrency=16, **({"temperature": 0.7, "extra_body": {"seed": seed}} if seed is not None else {}))
    dev = protocol.dev(T)
    out = await label(c, [Example(e.uid, e.state, None, None, e.group, {}) for e in dev], f"{T}:dev_{name}",
                      seeds=protocol.seeds(T) if v["seeds"] else None, guidance=(Path(__file__).parent / "guidance.txt").read_text() if v["guide"] else None)
    await c.aclose()
    tag = name + (f"_s{seed}" if seed is not None else "")
    save_jsonl(LD / f"dev_{tag}.jsonl", out)
    nf = sum(e.labels is None for e in out)
    for e in out:
        if e.labels is None: e.labels = {a: 3 for a in A}; e.meta = {}
    report(f"{tag} (fail {nf}, ${c.stats['cost']:.4f})", out, dev)

if __name__ == "__main__":
    for n in sys.argv[1:]:
        asyncio.run(run(n))
