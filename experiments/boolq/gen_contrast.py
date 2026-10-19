"""Contrast/paraphrase augmentation of the REAL pool questions (keeps BoolQ's question distribution):
for each pool (passage, question), the LLM writes paraphrases (same answer) and minimal edits that flip the
answer, each with its answer as determined by the passage."""
import asyncio, json, random, sys
from common import *

SYSTEM = """You create contrastive training data for a yes/no reading-comprehension classifier (BoolQ).
You get a Wikipedia PASSAGE and a real user QUESTION about it. Write {n} new questions:
- {k} paraphrases of the QUESTION that keep its meaning (and therefore its answer), worded differently, still sounding like a search query;
- {k} minimal edits of the QUESTION (change one entity, number, date, relation, quantifier or negation) chosen so that the passage gives the OPPOSITE answer to the edited question;
- {r} other questions about the same passage in the same style that need careful reading (one yes, one no when possible).
First decide the answer to the original QUESTION from the passage (the passage may imply it rather than state it; BoolQ answers follow what the passage supports, using common-sense inference). Then give the correct answer for every new question.
Reply with JSON only: {{"original": true|false, "qa": [{{"q": "<question>", "a": true|false}}, ...]}}"""


def build(state, k, r):
    n = 2 * k + r
    return [{"role": "system", "content": SYSTEM.format(n=n, k=k, r=r)},
            {"role": "user", "content": "PASSAGE:\n" + state["passage"].strip() + "\n\nQUESTION: " + state["question"].strip() + "\n\nJSON:"}]


async def main(model, tag, k, r, limit):
    c = client(model)
    gu = gold_uids()
    pool = [e for e in protocol.pool(TASK) if e.uid not in gu]
    random.Random(0).shuffle(pool)
    pool = pool[:limit] if limit else pool

    async def one(ex):
        try:
            raw = await c.complete_json(build(ex.state, k, r), purpose=f"boolq-contrast-{tag}", max_tokens=40 * (2 * k + r) + 80)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            return None, []
        if not isinstance(raw, dict):
            return None, []
        orig = raw.get("original") if isinstance(raw.get("original"), bool) else None
        out = []
        for j, it in enumerate(raw.get("qa") or []):
            if not isinstance(it, dict) or not isinstance(it.get("q"), str) or not isinstance(it.get("a"), bool) or len(it["q"].strip()) < 8:
                continue
            q = it["q"].strip(); q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
            out.append(Example(f"{ex.uid}-c{tag}-{j}", {"passage": ex.state["passage"], "question": q}, {QID: it["a"]}, None, "",
                               {"synthetic": True, "src_uid": ex.uid, "labeler": model, "kind": "contrast"}))
        return (ex.uid, orig), out

    res = await c.map(pool, one, desc=f"contrast:{tag}", progress=False)
    exs = [e for _, r_ in res for e in r_]
    origs = {o[0]: o[1] for o, _ in res if o and o[1] is not None}
    write_jsonl(DATA / f"contrast_{tag}.jsonl", exs)
    (DATA / f"contrast_{tag}_orig.json").write_text(json.dumps(origs))
    print(tag, "passages", len(pool), "questions", len(exs), "yes_rate", round(float(np.mean([e.labels[QID] for e in exs])), 3), f"cost=${c.stats['cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])))
