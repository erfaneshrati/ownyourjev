"""Sub-phrases of pool sentences (SST-style phrase-level transfer data) -> label with teacher."""
import asyncio, sys, re, json
sys.path.insert(0, "experiments/sst5")
from lab import *
n = int(sys.argv[1])
excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
pool = [e for e in P.pool(T) if e.uid not in excl]
rng = np.random.default_rng(7)
cands, seen = [], set()
for e in pool:
    s = e.state["sentence"]
    parts = re.split(r" (?:,|;|--|:|\.\.\.) | (?=but |although |though |yet |while |and )", s)
    if len(parts) < 2: continue
    for i, p in enumerate(parts):
        p = p.strip(" ,.;")
        w = p.split()
        if 3 <= len(w) <= 14 and p not in seen and p != s:
            seen.add(p); cands.append((e.uid, i, p))
print(len(cands), "candidate phrases", file=sys.stderr)
pick = [cands[i] for i in rng.permutation(len(cands))[:n]]
exs = [Example(f"{u}-ph{i}", {"sentence": p}, None, None, "", {}) for u, i, p in pick]
for x in exs[:15]: print(x.state["sentence"], file=sys.stderr)
g = open("experiments/sst5/guidance_v1.txt").read()
if len(sys.argv) > 2 and sys.argv[2] == "go":
    res = asyncio.run(run(f"phrases_v1_{n}", "deepseek/deepseek-v4.1-flash", exs, guidance=g))
    with open(OUT / f"phrases_v1_{n}_texts.jsonl", "w") as f:
        for x in exs: f.write(json.dumps({"uid": x.uid, "state": x.state}) + "\n")
