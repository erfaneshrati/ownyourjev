"""PAWS: compare labelers/prompts on dev (halves A=first 100, B=last 100). usage: p1_devlabel.py <variant...>"""
import sys
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
T = "paws"
GUIDE = """These pairs come from PAWS: sentence_2 was usually produced from sentence_1 by swapping or moving words, so word overlap is almost always very high and tells you nothing.
Decide by comparing the facts each sentence states, slot by slot: who does what to whom, which place/date/number belongs to which event, which adjective or modifier attaches to which noun, which entity a relative clause or prepositional phrase describes.
- If two names, places, dates, numbers or roles were swapped, or a modifier now attaches to a different noun, the meaning differs -> false.
- If only the syntax changed and every fact is preserved (clauses reordered, active/passive, a phrase moved without changing what it refers to, members of a symmetric list reordered, a synonym used) -> true.
- Ignore spacing, punctuation and tokenisation artefacts (e.g. " , ", " 's").
Check carefully before answering; small swaps are the whole point of this dataset."""
VARIANTS = {
    "flash_plain": dict(model="deepseek/deepseek-v4.1-flash", seeds=False, guide=False),
    "flash_seeded": dict(model="deepseek/deepseek-v4.1-flash", seeds=True, guide=False),
    "flash_seeded_guide": dict(model="deepseek/deepseek-v4.1-flash", seeds=True, guide=True),
    "pro_seeded": dict(model="deepseek/deepseek-v4-pro", seeds=True, guide=False),
    "pro_seeded_guide": dict(model="deepseek/deepseek-v4-pro", seeds=True, guide=True),
    "haiku_seeded_guide": dict(model="anthropic/claude-haiku-5.5", seeds=True, guide=True),
    "gemini3_seeded_guide": dict(model="google/gemini-3.1-flash-lite", seeds=True, guide=True),
    "flash_seeded_guide_soft": dict(model="deepseek/deepseek-v4.1-flash", seeds=True, guide=True, dist=True),
}
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
for v in sys.argv[1:]:
    c = VARIANTS[v]
    client = OpenRouterClient(model=c["model"])
    lab = run(label_examples(client, Q(T), [Example(e.uid, e.state, None, None) for e in dev], purpose=f"paws:dev:{v}",
                             seeds=protocol.seeds(T) if c["seeds"] else None, guidance=GUIDE if c["guide"] else None,
                             distribution=c.get("dist", False), progress=False))
    p = np.array([float(e.labels["answer"]) if e.labels else 0.5 for e in lab])
    if c.get("dist"):
        p = np.array([e.meta["soft"]["answer"] if e.labels else 0.5 for e in lab])
    np.save(ROOT / f"data/v2/paws/dev_{v}.npy", p)
    ok = (p >= 0.5) == y
    print(f"{v}: acc {ok.mean():.3f}  A {ok[:100].mean():.3f}  B {ok[100:].mean():.3f}  fail {sum(e.labels is None for e in lab)} predpos {(p>=0.5).sum()}", flush=True)
