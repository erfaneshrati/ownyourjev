"""Batched LLM labeling for banking77 (many queries per call; the long intent list is paid once per batch).
Labeler prompt uses rich intent descriptions (intents.py); option keys are unchanged."""
from __future__ import annotations

import asyncio, difflib, json, random, sys
from pathlib import Path

from ownyourjev.llm import OpenRouterClient  # noqa
from experiments.banking77.intents import DESC, GUIDANCE

OPTS = sorted(DESC)  # same set as the task options

def system_prompt(seeds=None, guidance=GUIDANCE, desc=DESC, soft=False):
    lines = ["You are a meticulous annotator for a bank's customer-support intent classifier.",
             "For each numbered customer query, choose the ONE intent key (exactly as written) that best describes it.", "",
             "## Intents"]
    for k in OPTS:
        lines.append(f"- {k}: {desc[k]}")
    lines += ["", "## Guidelines", guidance.strip(), ""]
    if seeds:
        lines.append("## Examples of correct labels")
        for s in seeds:
            lines.append(f"{json.dumps(s.state['query'])} -> {s.labels['answer']}")
        lines.append("")
    if soft:
        lines.append('## Output\nReturn a JSON object mapping each query number (as a string) to an object of up to 3 candidate intent keys with probabilities summing to 1, most likely first, e.g. {"1": {"card_arrival": 0.8, "card_delivery_estimate": 0.2}, "2": {...}}. JSON only.')
    else:
        lines.append('## Output\nReturn a JSON object mapping each query number (as a string) to its intent key, e.g. {"1": "card_arrival", "2": "...", ...}. JSON only.')
    return "\n".join(lines)


def _fix(k):
    if k in DESC:
        return k
    m = difflib.get_close_matches(str(k), OPTS, n=1, cutoff=0.8)
    return m[0] if m else None


async def label_texts(client, texts, *, purpose, seeds=None, guidance=GUIDANCE, desc=DESC, batch=20, soft=False, shuffle_seed=0):
    sysmsg = system_prompt(seeds, guidance, desc, soft)
    idx = list(range(len(texts)))
    random.Random(shuffle_seed).shuffle(idx)
    chunks = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    out = [None] * len(texts)

    async def run(chunk, attempt=0):
        user = "\n".join(f"{j+1}. {json.dumps(texts[i])}" for j, i in enumerate(chunk))
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "## Queries\n" + user + "\n\n## Answer (JSON only)"}]
        try:
            raw = await client.complete_json(msgs, purpose=purpose, max_tokens=(40 if soft else 16) * len(chunk) + 64)
        except Exception as e:  # noqa
            raw = {}
        missing = []
        for j, i in enumerate(chunk):
            v = raw.get(str(j + 1)) if isinstance(raw, dict) else None
            if soft and isinstance(v, dict):
                d = {}
                for k, p in v.items():
                    kk = _fix(k)
                    try:
                        if kk: d[kk] = d.get(kk, 0) + float(p)
                    except (TypeError, ValueError):
                        pass
                tot = sum(d.values())
                if tot > 0:
                    out[i] = {k: p / tot for k, p in d.items()}
                    continue
            elif not soft and v is not None and _fix(v):
                out[i] = _fix(v); continue
            missing.append(i)
        if missing and attempt < 2 and len(chunk) > 1:
            # retry missing ones in smaller groups
            await asyncio.gather(*[run(missing[k:k + 5], attempt + 1) for k in range(0, len(missing), 5)])
        elif missing and attempt < 2:
            await run(missing, attempt + 1) if False else None

    from tqdm.asyncio import tqdm_asyncio
    await tqdm_asyncio.gather(*[run(c) for c in chunks], desc=purpose)
    return out


def make_client(model, temperature=0.0, seed=None, **kw):
    eb = {"provider": {"sort": "throughput"}}
    if seed is not None:
        eb["seed"] = seed
    if "gpt-5" in model or "luna" in model:
        eb["reasoning"] = {"effort": "minimal"}
    return OpenRouterClient(model=model, temperature=temperature, extra_body=eb, **kw)
