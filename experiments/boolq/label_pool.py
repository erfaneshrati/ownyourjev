"""Label the whole boolq pool with a labeler (default qwen3.7-flash, soft labels)."""
import asyncio, sys
from common import *
from ownyourjev.label import label_examples


async def main(model, tag, dist, uids_file=None):
    c = client(model)
    gu = gold_uids()
    pool = [e for e in protocol.pool(TASK) if e.uid not in gu]
    if uids_file:
        want = set(Path(uids_file).read_text().split())
        pool = [e for e in pool if e.uid in want]
    out = await label_examples(c, Q, pool, purpose=f"boolq-pool-{tag}", distribution=dist, progress=False)
    write_jsonl(DATA / f"pool_{tag}.jsonl", out)
    print(tag, len(out), "invalid", sum(e.labels is None for e in out), f"cost=${c.stats['cost']:.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] if len(sys.argv) > 4 else None))
