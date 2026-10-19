"""Measure labeler agreement with gold on dev (halves A/B reported separately)."""
import asyncio, json, sys
from common import *
from ownyourjev.label import label_examples

GUIDE = Path(__file__).with_name("guidance.txt")


async def run(model, use_seeds, guide, distribution, tag):
    c = client(model)
    dev = protocol.dev(TASK)
    seeds = protocol.seeds(TASK) if use_seeds else None
    g = GUIDE.read_text() if guide else None
    out = await label_examples(c, Q, dev, purpose=f"boolq-dev-{tag}", seeds=seeds, guidance=g, distribution=distribution, progress=False)
    write_jsonl(DATA / "dev_labels" / f"{tag}.jsonl", out)
    A, B = dev_halves()
    res = {}
    for name, part in [("A", out[:100]), ("B", out[100:]), ("all", out)]:
        ok = [e for e in part if e.labels]
        res[name] = round(acc([e.gold[QID] for e in ok], [e.labels[QID] for e in ok]), 3)
    res["invalid"] = sum(e.labels is None for e in out)
    res["yes_rate"] = round(float(np.mean([e.labels[QID] for e in out if e.labels])), 3)
    print(tag, json.dumps(res), f"cost=${c.stats['cost']:.4f}", flush=True)


if __name__ == "__main__":
    model, flags = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    tag = model.split("/")[-1] + ("-seeds" if "s" in flags else "") + ("-guide" if "g" in flags else "") + ("-dist" if "d" in flags else "")
    asyncio.run(run(model, "s" in flags, "g" in flags, "d" in flags, tag))
