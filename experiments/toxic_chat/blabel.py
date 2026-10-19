"""Batched labeling: one system prompt (questions + guidance + seeds), N prompts per request, soft answers.
usage: blabel.py dev|pool NAME MODEL BS [GUIDANCE] [temperature] [seed]"""
import asyncio, json, sys
from dataclasses import replace
import numpy as np
from common import *
import prompts
from ownyourjev.llm import OpenRouterClient
from ownyourjev.label import SYSTEM_PREAMBLE

MAXC = 1500  # chars per prompt shown to the labeler


def system_prompt(guidance, seeds):
    L = [SYSTEM_PREAMBLE, "", "Each STATE is a user `prompt` sent to an AI chatbot. Answer two questions for every prompt:",
         f"- toxic: {Q['toxic'].instructions}", f"- jailbreak: {Q['jailbreak'].instructions}", "", "## Labeling guidelines", guidance.strip(), ""]
    if seeds:
        L.append("## Examples of correct answers")
        for s in seeds:
            L.append(json.dumps(s.state["prompt"][:400]) + " -> " + json.dumps(s.labels))
        L.append("")
    L.append('## Output\nYou get numbered prompts. Reply with ONE JSON object mapping each number (string) to {"toxic": <probability 0-1 that the answer is true>, "jailbreak": <probability 0-1>}. JSON only.')
    return "\n".join(L)


async def _batch(cli, sysmsg, batch, purpose, extra_tok=0):
    user = "\n\n".join(f"### {i + 1}\n{json.dumps(e.state['prompt'][:MAXC] + (' [...truncated]' if len(e.state['prompt']) > MAXC else ''))}" for i, e in enumerate(batch))
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    try:
        raw = await cli.complete_json(msgs, purpose=purpose, max_tokens=30 * len(batch) + 60 + extra_tok)
    except Exception:
        raw = {}
    out = []
    for i in range(len(batch)):
        r = raw.get(str(i + 1)) if isinstance(raw, dict) else None
        try:
            out.append({h: min(1.0, max(0.0, float(r[h]))) for h in HEADS})
        except Exception:
            out.append(None)
    return out


async def label(exs, model, bs, guidance, purpose, temperature=0.0, seed=None, use_seeds=True, reasoning=None):
    eb = {"seed": seed} if seed is not None else {}
    if not model.startswith(("deepseek/deepseek-v4.1-flash", "qwen")):
        eb["provider"] = {"sort": "throughput"}
    extra_tok = 0
    if reasoning:
        if reasoning.startswith("max"):
            eb["reasoning"] = {"max_tokens": int(reasoning[3:])}; extra_tok = int(reasoning[3:])
        else:
            eb["reasoning"] = {"enabled": True, "effort": reasoning}; extra_tok = 6000
    cli = OpenRouterClient(model=model, concurrency=24, temperature=temperature, extra_body=eb or None)
    sysmsg = system_prompt(guidance, protocol.seeds(T) if use_seeds else None)
    batches = [exs[i:i + bs] for i in range(0, len(exs), bs)]
    res = await asyncio.gather(*[_batch(cli, sysmsg, b, purpose, extra_tok) for b in batches])
    labs = [x for r in res for x in r]
    miss = [i for i, l in enumerate(labs) if l is None]
    if miss:
        rr = await asyncio.gather(*[_batch(cli, sysmsg, [exs[i]], purpose + "-retry", extra_tok) for i in miss])
        for i, r in zip(miss, rr):
            labs[i] = r[0]
    print("stats", cli.stats, "missing", sum(l is None for l in labs))
    return [replace(e, labels=({h: l[h] >= .5 for h in HEADS} if l else None), meta={**e.meta, "labeler": model, "soft": l} if l else {**e.meta}) for e, l in zip(exs, labs)]


if __name__ == "__main__":
    split, name, model, bs = sys.argv[1:5]
    g = getattr(prompts, sys.argv[5]) if len(sys.argv) > 5 else prompts.GUIDANCE_V2
    temp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    seed = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "-" else None
    reasoning = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != "-" else None
    use_seeds = not (len(sys.argv) > 9 and sys.argv[9] == "noseeds")
    exs = protocol.dev(T) if split == "dev" else protocol.pool(T)
    if len(sys.argv) > 10:
        keep = set(open(sys.argv[10]).read().split()); exs = [e for e in exs if e.uid in keep]
    out = asyncio.run(label(exs, model, int(bs), g, f"tc:b{split}:{name}", temp, seed, use_seeds, reasoning))
    save_jsonl(D / f"{split}_{name}.jsonl", out)
    if split == "dev":
        for h in HEADS:
            y = [e.gold[h] for e in exs]; p = np.nan_to_num([soft(o, h) if o.labels else np.nan for o in out])
            print(report(y, p, f"{name}:{h}"), f"cv={cv_threshold_f1(y, p):.3f}")
