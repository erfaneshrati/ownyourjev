"""Batch labeling: N sentences per call, soft distribution per sentence (percentages)."""
import asyncio, sys, json
sys.path.insert(0, "experiments/sst5")
from lab import *
from ownyourjev.label import build_messages

def sys_prompt(guidance):
    m = build_messages(Q, {"sentence": "x"}, seeds=SEEDS, guidance=guidance, distribution=True)
    s = m[0]["content"]
    s += ("\n\n## Batch mode\nYou will receive several numbered snippets. Rate each one independently. Return one JSON object "
          'mapping each snippet number to its probability list over levels 0-4 in percent (integers summing to 100), e.g. '
          '{"1": [5, 70, 20, 5, 0], "2": [0, 0, 10, 50, 40]}. Cover every number.')
    return s

async def label_batch(name, model, exs, guidance=None, bs=20, pricey=False, temperature=0.0, seed=None):
    c = client(model, temperature, seed, pricey); sp = sys_prompt(guidance)
    chunks = [exs[i:i+bs] for i in range(0, len(exs), bs)]
    async def one(ch):
        user = "## SNIPPETS\n" + "\n".join(f'{i+1}. {e.state["sentence"]}' for i, e in enumerate(ch)) + "\n\n## ANSWER (JSON only)"
        for attempt in range(2):
            try:
                raw = await c.complete_json([{"role": "system", "content": sp}, {"role": "user", "content": user + ("" if attempt == 0 else " ")}], purpose=f"sst5:{name}", max_tokens=40 + 30 * len(ch))
                out = {}
                for i, e in enumerate(ch):
                    v = raw.get(str(i + 1))
                    if v is None or len(v) != 5: continue
                    v = np.clip(np.array(v, float), 0, None)
                    if v.sum() <= 0: continue
                    out[e.uid] = (v / v.sum()).tolist()
                if len(out) >= len(ch) * 0.8 or attempt == 1: return out
            except Exception as ex:
                err = ex
        return {}
    res = await c.map(chunks, one, progress=False)
    await c.aclose()
    merged = {k: v for d in res for k, v in d.items()}
    with open(OUT / f"{name}.jsonl", "w") as f:
        for e in exs:
            s = merged.get(e.uid)
            f.write(json.dumps({"uid": e.uid, "labels": {"answer": int(np.argmax(s))} if s else None, "soft": s}) + "\n")
    print(name, "cost", round(c.stats["cost"], 4), "calls", c.stats["calls"], "labeled", len(merged), "/", len(exs), file=sys.stderr)

if __name__ == "__main__":
    tag, model, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    g = None if tag == "base" else open(f"experiments/sst5/guidance_{tag}.txt").read()
    which = sys.argv[4] if len(sys.argv) > 4 else "dev"
    if which == "dev":
        name = f"devB{bs}_{tag}_{model.split('/')[1]}"
        asyncio.run(label_batch(name, model, DEV, g, bs, pricey=True)); report(name)
    else:
        excl = {e.uid for e in DEV} | {e.uid for e in SEEDS}
        pool = [e for e in P.pool(T) if e.uid not in excl]
        name = f"poolB{bs}_{tag}_{model.split('/')[1]}"
        asyncio.run(label_batch(name, model, pool, g, bs, pricey=True))
