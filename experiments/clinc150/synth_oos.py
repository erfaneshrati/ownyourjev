"""Generate out-of-scope utterances (near-domain and far-domain) with an LLM, then keep only those the
labeler also calls oos. Writes data/v2/clinc150/synth_oos.jsonl"""
import asyncio, json, random, sys
from pathlib import Path
from ownyourjev.data import Example, write_jsonl
from ownyourjev.llm.client import OpenRouterClient
from benchmarks.v2 import protocol as P
from experiments.clinc150.intents import DESC
from experiments.clinc150 import blabel

TOPICS = [
    "banking and personal finance (but not the covered intents: e.g. stocks, mortgages, loans, investing advice, crypto)",
    "credit cards and shopping (things not covered: e.g. store hours, coupons, product comparisons, returns)",
    "cooking and food (not covered: e.g. wine pairing, kitchen equipment, diets, food trivia, grocery prices)",
    "home and family (not covered: e.g. cleaning tips, pets care, gardening, home repair, parenting, chores)",
    "cars and driving (not covered: e.g. car features, blind spot, car buying, parking, speeding tickets, road rules)",
    "travel (not covered: e.g. tourist attractions history, packing advice, airport lounges, cruises, train tickets)",
    "work and office (not covered: e.g. HR policies, office supplies, coworkers, job search, salary negotiation)",
    "the assistant itself and device settings (not covered: e.g. privacy settings, battery, wifi, apps, screen brightness, updates)",
    "sports, games and entertainment (scores, schedules, movies, tv shows, celebrities, video games, books)",
    "health, medicine and fitness (symptoms, doctors, exercise, sleep, mental health)",
    "science, history, geography and general knowledge questions",
    "music theory, art, hobbies, and creative requests",
    "news, politics, law, religion, astrology/horoscopes",
    "personal opinions, philosophical questions, relationship advice, random chit-chat requests",
    "local services and errands (plumbers, hair salons, pharmacies, post office, dry cleaning)",
    "education and school (homework help, exams, colleges, learning languages)",
]
STYLES = ["short command", "question", "casual with typos", "terse fragment", "polite full sentence", "indirect request"]


async def main(n_per_call=30, reps=6):
    client = OpenRouterClient(temperature=1.0, concurrency=16)
    sysmsg = ("You write realistic utterances that users say to a voice/virtual assistant. Reply with JSON only.\n"
              "The assistant supports ONLY these intents:\n" + ", ".join(k for k in DESC if k != "oos") +
              "\nYou must write utterances that are OUT OF SCOPE: they must NOT be a request for any of the supported intents, "
              "though many should be in a nearby domain so they look plausible. Style: lowercase, short (3-15 words) like real assistant queries.")
    jobs = []
    for t in TOPICS:
        for r in range(reps):
            style = STYLES[r % len(STYLES)]
            user = f"Topic: {t}\nStyle: {style}\nWrite {n_per_call} distinct out-of-scope utterances. Return {{\"utterances\": [\"...\", ...]}}"
            jobs.append((t, r, [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]))

    async def one(job):
        t, r, msgs = job
        c = OpenRouterClient(temperature=1.0, extra_body={"seed": r}, concurrency=1)
        try:
            raw = await client.complete_json(msgs, purpose="clinc-synth-oos", max_tokens=1200) if r == 0 else await c.complete_json(msgs, purpose="clinc-synth-oos", max_tokens=1200)
        except Exception as e:
            print("fail", e); return []
        finally:
            await c.aclose()
        us = raw.get("utterances", []) if isinstance(raw, dict) else []
        return [u.strip().lower() for u in us if isinstance(u, str) and 2 <= len(u.split()) <= 25]

    res = await asyncio.gather(*[one(j) for j in jobs])
    seen, exs = set(), []
    for (t, r, _), us in zip(jobs, res):
        for u in us:
            if u not in seen:
                seen.add(u)
                exs.append(Example(f"synoos-{len(exs)}", {"utterance": u}, None, None, "", {"topic": t}))
    print("generated", len(exs), client.stats)
    await client.aclose()
    lab = await blabel.label(exs, seeds=P.seeds("clinc150"), purpose="clinc-synth-oos-filter")
    keep = [e for e in lab if e.labels and e.labels["answer"] == "oos"]
    print("kept as oos", len(keep), "of", len(lab))
    out = Path("data/v2/clinc150"); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "synth_oos_all.jsonl", lab)
    write_jsonl(out / "synth_oos.jsonl", keep)

asyncio.run(main())
