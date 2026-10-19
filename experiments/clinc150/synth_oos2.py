"""Round 2 synthetic oos: near-miss requests per CLINC domain (shown that domain's intents), filtered by the labeler.
Writes data/v2/clinc150/synth_oos2.jsonl"""
import asyncio
from pathlib import Path
from ownyourjev.data import Example, write_jsonl, read_jsonl
from ownyourjev.llm.client import OpenRouterClient
from benchmarks.v2 import protocol as P
from experiments.clinc150.intents import DESC
from experiments.clinc150 import blabel

keys = [k for k in DESC if k != "oos"]
# domain blocks follow the order of DESC (banking..small talk)
DOM = {
    "banking and work/HR": keys[0:27], "credit cards": keys[27:42], "kitchen and dining": keys[42:57],
    "home (lists, reminders, calendar, music, smart home)": keys[57:76], "auto and commute": keys[76:92],
    "travel": keys[92:107], "utility (time, math, facts, calls, meetings)": keys[107:123],
    "assistant settings and meta": keys[123:140], "small talk with the assistant": keys[140:],
}
FAR = ["sports scores and schedules", "movies, tv and celebrities", "medical symptoms and health", "science and trivia questions",
       "stock market and investing", "shopping for products and store info", "pets and animals", "home repair and cleaning",
       "phone/computer tech support", "politics, news and law", "games, puzzles and riddles", "personal advice and feelings"]


async def main():
    jobs = []
    for d, ks in DOM.items():
        block = "\n".join(f"- {k}: {DESC[k]}" for k in ks)
        for r in range(5):
            jobs.append((r, f"Domain: {d}. The assistant supports ONLY these intents in this domain:\n{block}\n"
                         "Write 30 distinct utterances a user might say to the assistant in this same domain that are NOT any of the "
                         "supported intents (near misses: related topic, but a different need the assistant does not support). "
                         "Lowercase, 3-15 words, varied phrasing (questions, commands, casual, typos)."))
    for t in FAR:
        for r in range(2):
            jobs.append((r, f"Topic: {t}. Write 30 distinct short utterances (lowercase, 3-15 words) a user might say to a virtual assistant about this topic. "
                         "None of them may be a request for these supported intents: " + ", ".join(keys)))
    sysmsg = "You write realistic, diverse user utterances for a virtual assistant dataset. Reply with JSON only: {\"utterances\": [\"...\", ...]}"

    async def one(job):
        r, user = job
        c = OpenRouterClient(temperature=1.0, extra_body={"seed": 100 + r}, concurrency=1)
        try:
            raw = await c.complete_json([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], purpose="clinc-synth-oos2", max_tokens=1200)
        except Exception as e:
            print("fail", e); return []
        finally:
            await c.aclose()
        us = raw.get("utterances", []) if isinstance(raw, dict) else []
        return [u.strip().lower() for u in us if isinstance(u, str) and 2 <= len(u.split()) <= 25]

    res = await asyncio.gather(*[one(j) for j in jobs])
    seen = {e.state["utterance"] for e in read_jsonl("data/v2/clinc150/synth_oos_all.jsonl")}
    exs = []
    for us in res:
        for u in us:
            if u not in seen:
                seen.add(u); exs.append(Example(f"synoos2-{len(exs)}", {"utterance": u}, None, None, "", {}))
    print("generated", len(exs))
    lab = await blabel.label(exs, seeds=P.seeds("clinc150"), purpose="clinc-synth-oos2-filter")
    keep = [e for e in lab if e.labels and e.labels["answer"] == "oos"]
    print("kept as oos", len(keep), "of", len(lab))
    write_jsonl("data/v2/clinc150/synth_oos2_all.jsonl", lab)
    write_jsonl("data/v2/clinc150/synth_oos2.jsonl", keep)

asyncio.run(main())
