"""Round 2 data: LLM-generated partner sentences at controlled similarity levels, steered by pool sentences.
For each source sentence the generator writes 5 partners targeting SemEval levels 1..5 (same style/register as the
source). The target level is NOT used as a label: all pairs are re-scored by the 4-labeler ensemble."""
import asyncio, json, random, sys
from common import *
from ownyourjev.llm import OpenRouterClient
N_SRC = int(sys.argv[1]); MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-v4.1-flash"
SYS = """You create training data for semantic textual similarity (SemEval STS, 0-5 scale). For each numbered SOURCE sentence, write five new sentences, one for each target similarity level to the source:
- "1": same general topic only, different event/subject (e.g. a different news story in the same domain).
- "2": not equivalent but shares some details (same actor or same kind of event, key elements differ).
- "3": roughly equivalent but some important information differs or is missing (different number, extra/missing substantial clause, related but different object).
- "4": mostly equivalent, only unimportant details differ (a modifier added/removed, minor rewording plus one small detail).
- "5": a full paraphrase with the same meaning (reword, do not just copy).
Match the source's style exactly: news headlines stay terse headlines (keep casing conventions), image captions stay simple captions ("A man is ..."), forum/QA text keeps its register. Keep a similar length. Vary the kind of change; for some level-2/3 items use a negation, a different entity, a different number or a swapped role.
Return one JSON object: {"<source number>": {"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}, ...}"""
async def main():
    excl = {s.strip() for e in protocol.dev(T) + protocol.seeds(T) for s in e.state.values()}
    pool = protocol.pool(T); rng = random.Random(1); rng.shuffle(pool)
    srcs = []
    for e in pool:
        s = e.state[rng.choice(["sentence_1", "sentence_2"])].strip()
        if s not in excl and s not in srcs: srcs.append(s)
        if len(srcs) >= N_SRC: break
    cl = OpenRouterClient(model=MODEL, temperature=0.7, concurrency=32)
    B = 10; batches = [srcs[i:i+B] for i in range(0, len(srcs), B)]
    async def one(b):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "\n".join(f"SOURCE {i}: {s}" for i, s in enumerate(b, 1)) + "\n\nJSON only."}]
        try:
            raw = await cl.complete_json(msgs, purpose="stsb:r2:gen", max_tokens=2500)
        except Exception as ex:
            return []
        out = []
        for i, s in enumerate(b, 1):
            v = raw.get(str(i)) if isinstance(raw, dict) else None
            if not isinstance(v, dict): continue
            for lvl, t in v.items():
                if isinstance(t, str) and t.strip() and t.strip() != s:
                    pair = (s, t.strip()) if rng.random() < 0.5 else (t.strip(), s)
                    out.append({"state": {"sentence_1": pair[0], "sentence_2": pair[1]}, "target": int(lvl) if str(lvl).isdigit() else None})
        return out
    res = await cl.map(batches, one, desc="gen")
    rows = [r for b in res for r in b]
    with open(D / "new_gen.jsonl", "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"uid": f"gen-{k}", **r}) + "\n")
    print("generated", len(rows), "stats", cl.stats)
asyncio.run(main())
