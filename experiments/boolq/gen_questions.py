"""Generate extra yes/no questions (with answers) for pool passages: a 5-10x data multiplier.

Each pool passage gets N new questions in BoolQ style (natural search queries), balanced yes/no, including hard
negatives that reuse passage words and questions that need inference. The generator's own answer is the label.
"""
import asyncio, json, random, sys
from common import *
from ownyourjev.spec import canonical_json

SYSTEM = """You write training data for a reading-comprehension classifier (BoolQ style).
Given a Wikipedia PASSAGE, write {n} new yes/no questions that a curious person might type into a search engine, each answerable from the passage alone, with the correct answer.

Style: like real search queries, e.g.
{examples}

Requirements:
- Exactly {half} questions with answer true (yes) and {half} with answer false (no), in random order.
- Mix difficulty: a few direct facts; several that need paraphrase, combining two sentences, or inference (numbers, dates, comparisons, part-whole, cause, exceptions); hard "no" questions that reuse words and entities from the passage but swap an entity, attribute, number, time, or relation so the answer is no.
- The passage must clearly determine every answer (no guessing from world knowledge, no unanswerable questions). Do not ask about the passage itself ("does the passage say...").
- Vary question openings (is/are/does/do/did/was/can/will/has...); keep each question short.
Reply with JSON only: {{"qa": [{{"q": "<question>", "a": true|false}}, ...]}}"""


def build(passage, n, seed_qs):
    half = n // 2
    sys_msg = SYSTEM.format(n=n, half=half, examples="\n".join(f"- {q}" for q in seed_qs))
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": "PASSAGE:\n" + passage.strip() + "\n\nJSON:"}]


async def main(model, tag, n, limit):
    c = client(model)
    gu = gold_uids()
    seed_qs = [e.state["question"] for e in protocol.seeds(TASK)]
    pool = [e for e in protocol.pool(TASK) if e.uid not in gu]
    random.Random(0).shuffle(pool)
    pool = pool[:limit] if limit else pool

    async def one(ex):
        try:
            raw = await c.complete_json(build(ex.state["passage"], n, seed_qs), purpose=f"boolq-gen-{tag}", max_tokens=40 * n + 60)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            return []
        items = raw.get("qa") if isinstance(raw, dict) else raw
        out = []
        for j, it in enumerate(items if isinstance(items, list) else []):
            if not isinstance(it, dict) or not isinstance(it.get("q"), str) or not isinstance(it.get("a"), bool):
                continue
            q = it["q"].strip()
            if len(q) < 8:
                continue
            q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
            out.append(Example(f"{ex.uid}-g{tag}-{j}", {"passage": ex.state["passage"], "question": q}, {QID: it["a"]}, None, "",
                               {"synthetic": True, "src_uid": ex.uid, "labeler": model}))
        return out

    res = await c.map(pool, one, desc=f"gen:{tag}", progress=False)
    exs = [e for r in res for e in r]
    write_jsonl(DATA / f"gen_{tag}.jsonl", exs)
    print(tag, "passages", len(pool), "questions", len(exs), "yes_rate", round(float(np.mean([e.labels[QID] for e in exs])), 3), f"cost=${c.stats['cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
