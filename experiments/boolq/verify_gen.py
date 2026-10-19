"""Check generated answers against an independent labeler on a sample."""
import asyncio, random, sys
from common import *
from ownyourjev.label import label_examples


async def main(tag, model, n):
    exs = read_jsonl(DATA / f"gen_{tag}.jsonl")
    random.Random(0).shuffle(exs)
    exs = exs[:n]
    c = client(model)
    out = await label_examples(c, Q, exs, purpose=f"boolq-verify-{tag}", progress=False)
    ok = [(a, b) for a, b in zip(exs, out) if b.labels]
    agree = np.mean([a.labels[QID] == b.labels[QID] for a, b in ok])
    yes = [a.labels[QID] == b.labels[QID] for a, b in ok if a.labels[QID]]
    no = [a.labels[QID] == b.labels[QID] for a, b in ok if not a.labels[QID]]
    print(tag, model, f"agree={agree:.3f} on_yes={np.mean(yes):.3f} on_no={np.mean(no):.3f}", f"cost=${c.stats['cost']:.4f}")
    for a, b in ok:
        if a.labels[QID] != b.labels[QID]:
            print("  gen=", a.labels[QID], "|", a.state["question"])


asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
