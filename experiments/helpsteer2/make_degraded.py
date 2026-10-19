"""Rule-based degraded variants of labeled pool responses with known quality direction (no LLM cost).
mismatch: response paired with another prompt -> helpfulness/correctness 0. truncate: response cut to 25-45% -> lower."""
import random, sys
from common import *
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(0)
hold = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
ex = [e for e in load_jsonl(LD / src) if e.labels and e.uid not in hold]
rng.shuffle(ex)
res = []
for i, e in enumerate(ex[:n]):
    o = ex[(i + n) % len(ex)]
    if o.state["prompt"] == e.state["prompt"]:
        continue
    l = dict(e.labels)
    res.append(Example(e.uid + "-mis", {"prompt": e.state["prompt"], "response": o.state["response"]},
                       {"helpfulness": 0, "correctness": 0, "coherence": o.labels["coherence"], "complexity": o.labels["complexity"], "verbosity": o.labels["verbosity"]}, None, "", {"synthetic": "mismatch"}))
for e in ex[n:2 * n]:
    r = e.state["response"]
    if len(r) < 400:
        continue
    cut = r[: int(len(r) * rng.uniform(0.25, 0.45))]
    l = e.labels
    res.append(Example(e.uid + "-trunc", {"prompt": e.state["prompt"], "response": cut},
                       {"helpfulness": max(0, l["helpfulness"] - 2), "correctness": max(0, l["correctness"] - 2), "coherence": max(0, l["coherence"] - 1),
                        "complexity": l["complexity"], "verbosity": max(0, l["verbosity"] - 1)}, None, "", {"synthetic": "truncate"}))
save_jsonl(LD / out, res)
print(len(res))
