"""Average soft label distributions of several label files (self-consistency). usage: merge_votes.py <out_name> <file1> <file2> ..."""
import sys
import numpy as np
from common import *
from ownyourjev.data import read_jsonl, write_jsonl, Example
out, files = sys.argv[1], sys.argv[2:]
runs = [{e.uid: e for e in read_jsonl(DATA / f"{f}.jsonl")} for f in files]
res = []
for uid, e in runs[0].items():
    softs = [r[uid].meta["soft"] for r in runs if uid in r and r[uid].labels]
    if not softs:
        res.append(e); continue
    soft = {d: list(np.mean([s[d] for s in softs], 0)) for d in DIMS}
    labels = {d: int(np.argmax(soft[d])) for d in DIMS}
    res.append(Example(e.uid, e.state, labels, e.gold, e.group, {**e.meta, "soft": soft, "n_votes": len(softs)}))
write_jsonl(DATA / f"{out}.jsonl", res); print(out, len(res))
