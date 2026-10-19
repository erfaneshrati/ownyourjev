"""Round 2 (b): BoolQ-like synthetic questions. Step 1 writes search queries from the TOPIC only (first words of
the passage), in the style of the 20 seed questions, without seeing the passage. Step 2 answers them from the
full passage with two labelers (yes / no / unknown), one call per passage; keep items where both agree and
neither says unknown."""
import asyncio, json, random, re, sys
from common import *

Q_SYS = """You imitate real Google search queries that are yes/no questions (the BoolQ dataset).
People type these BEFORE reading any article, from what they already half-know or wonder about a topic.
Real examples:
{examples}
Given only a TOPIC hint, write {n} different yes/no queries a curious person might search about it. Mix:
facts they are unsure about, common misconceptions, "is X the same as Y", "can you / is it legal / does X have Y",
"is X based on a true story", "will there be a season N", comparisons, membership ("is X part of Y"). Use
lowercase-ish casual query wording like the examples. Do not assume the answer; aim for roughly half yes, half no.
Reply with JSON only: {{"queries": ["...", ...]}}"""

A_SYS = """You answer yes/no questions using a Wikipedia PASSAGE, like a BoolQ annotator: answer "yes" or "no" when
the passage states or clearly implies the answer (reasonable common-sense inference allowed); answer "unknown"
when the passage does not let you decide.
Reply with JSON only: {"answers": ["yes"|"no"|"unknown", ...]} (one per question, same order)."""


def topic_hint(passage: str, words: int = 12) -> str:
    return " ".join(passage.split()[:words]) + " ..."


async def main(tag, n_q, limit, offset=0):
    gen = client("qwen/qwen3.7-flash")
    import os
    labs = [client(m) for m in os.environ.get("REALQ_LABELERS", "qwen/qwen3.7-flash,deepseek/deepseek-v4.1-flash").split(",")]
    gu = gold_uids()
    seed_qs = [e.state["question"] for e in protocol.seeds(TASK)]
    pool = [e for e in protocol.pool(TASK) if e.uid not in gu]
    random.Random(7).shuffle(pool)
    pool = pool[offset:offset + limit]
    qsys = Q_SYS.format(examples="\n".join(f"- {q}" for q in seed_qs), n=n_q)

    async def one(ex):
        try:
            raw = await gen.complete_json([{"role": "system", "content": qsys}, {"role": "user", "content": "TOPIC: " + topic_hint(ex.state["passage"]) + "\nJSON:"}],
                                          purpose=f"boolq-realq-gen-{tag}", max_tokens=30 * n_q + 40)
            qs = [q.strip() for q in (raw.get("queries") or []) if isinstance(q, str) and len(q.strip()) > 8][:n_q]
        except (RuntimeError, ValueError, json.JSONDecodeError, AttributeError):
            return []
        if not qs:
            return []
        qs = [q[0].upper() + q[1:] + ("" if q.endswith("?") else "?") for q in qs]
        user = "PASSAGE:\n" + ex.state["passage"].strip() + "\n\nQUESTIONS:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs)) + "\n\nJSON:"
        answers = []
        for c in labs:
            try:
                r = await c.complete_json([{"role": "system", "content": A_SYS}, {"role": "user", "content": user}], purpose=f"boolq-realq-ans-{tag}", max_tokens=12 * len(qs) + 30)
                a = r.get("answers") if isinstance(r, dict) else None
                answers.append(a if isinstance(a, list) and len(a) == len(qs) else None)
            except (RuntimeError, ValueError, json.JSONDecodeError):
                answers.append(None)
        out = []
        for j, q in enumerate(qs):
            av = [str(a[j]).strip().lower() if a else None for a in answers]
            rec = {"q": q, "votes": av}
            ok = all(v in ("yes", "no") for v in av) and len(set(av)) == 1
            out.append(Example(f"{ex.uid}-r{tag}-{j}", {"passage": ex.state["passage"], "question": q}, {QID: av[0] == "yes"} if ok else None, None, "",
                               {"synthetic": True, "src_uid": ex.uid, "kind": "realq", "votes": av}))
        return out

    res = await gen.map(pool, one, desc=f"realq:{tag}", progress=False)
    exs = [e for r in res for e in r]
    write_jsonl(DATA / f"realq_{tag}.jsonl", exs)
    kept = [e for e in exs if e.labels]
    unk = sum(any(v == "unknown" for v in e.meta["votes"]) for e in exs)
    cost = gen.stats["cost"] + sum(c.stats["cost"] for c in labs)
    print(tag, "passages", len(pool), "questions", len(exs), "kept", len(kept), "unknown", unk,
          "yes_rate_kept", round(float(np.mean([e.labels[QID] for e in kept])), 3) if kept else None, f"cost=${cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0))
