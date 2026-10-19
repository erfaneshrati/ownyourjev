"""Batched LLM labeling: one system prompt (intent list + descriptions + rules + seeds), N utterances per request.
150 options make the prompt ~3k tokens, so batching cuts label cost ~20x vs ownyourjev.label.label_examples."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from ownyourjev.data import Example
from ownyourjev.llm.client import OpenRouterClient
from experiments.clinc150.intents import DESC

GUIDE_V1 = """- Pick the single intent that best matches what the user wants the assistant to do or answer.
- Utterances are short, lowercase, may have typos; paraphrases and indirect wording still count for an intent.
- Read vs. update: checking/reading a list, reminder, calendar or schedule -> the plain intent (shopping_list, todo_list, reminder, calendar, meeting_schedule); adding/removing/changing/creating -> the *_update intent (or schedule_meeting / reminder_update / calendar_update).
- freeze_account = user wants the account frozen; account_blocked = account is already blocked/locked and the user asks why or reports it.
- distance = how far/how long between places; directions = route/how to get there.
- travel_notification = tell the bank about travel; travel_alert = safety warnings about a destination.
- oos (out of scope) = the request is about something none of the intents cover, even if it is in a related domain (e.g. stock prices, sports results, horoscopes, car blind spot, music theory, privacy settings). Do NOT use oos for a request that is a paraphrase of an existing intent."""


def system_prompt(guide: str = GUIDE_V1, seeds: list[Example] | None = None, with_desc: bool = True) -> str:
    lines = [
        "You are a meticulous annotator of user utterances sent to a virtual assistant (banking, credit cards, kitchen, home, auto, travel, utility, work, small talk, assistant settings).",
        "Task: Which intent does the user's `utterance` to a virtual assistant express? Choose exactly one intent key from the list.",
        "",
        "## Intents (key: description)",
    ]
    for k, v in DESC.items():
        lines.append(f"- {k}: {v}" if with_desc else f"- {k}")
    lines += ["", "## Rules", guide.strip()]
    if seeds:
        lines += ["", "## Labeled examples"]
        for s in seeds:
            lines.append(f"{json.dumps(s.state['utterance'])} -> {s.labels['answer']}")
    lines += ["", "## Output", 'You get numbered utterances. Reply with one JSON object mapping each number (as a string) to an intent key, e.g. {"1": "balance", "2": "oos"}. JSON only.']
    return "\n".join(lines)


async def _label_batch(client, sysmsg, batch, purpose):
    user = "\n".join(f"{i + 1}. {json.dumps(e.state['utterance'])}" for i, e in enumerate(batch))
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    try:
        raw = await client.complete_json(msgs, purpose=purpose, max_tokens=24 * len(batch) + 40)
    except Exception as e:  # noqa
        raw = {}
    out = []
    for i, e in enumerate(batch):
        lab = raw.get(str(i + 1)) if isinstance(raw, dict) else None
        if isinstance(lab, str):
            lab = lab.strip()
        out.append(lab if lab in DESC else None)
    return out


async def label(examples: list[Example], *, model="deepseek/deepseek-v4.1-flash", bs=20, guide=GUIDE_V1, seeds=None,
                with_desc=True, purpose="clinc-blabel", concurrency=16, extra_body=None, temperature=0.0) -> list[Example]:
    client = OpenRouterClient(model=model, concurrency=concurrency, extra_body=extra_body, temperature=temperature)
    sysmsg = system_prompt(guide, seeds, with_desc)
    batches = [examples[i:i + bs] for i in range(0, len(examples), bs)]
    from tqdm.asyncio import tqdm_asyncio
    res = await tqdm_asyncio.gather(*[_label_batch(client, sysmsg, b, purpose) for b in batches], desc=purpose)
    labs = [l for r in res for l in r]
    # retry failures one by one in small batches
    miss = [i for i, l in enumerate(labs) if l is None]
    if miss:
        rr = await asyncio.gather(*[_label_batch(client, sysmsg, [examples[i]], purpose + "-retry") for i in miss])
        for i, r in zip(miss, rr):
            labs[i] = r[0]
    print("stats", client.stats, "missing", sum(l is None for l in labs))
    await client.aclose()
    return [replace(e, labels=({"answer": l} if l else None), meta={**e.meta, "labeler": model}) for e, l in zip(examples, labs)]
