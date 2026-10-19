"""Round 2: stronger labelers for helpfulness/correctness only (2-question prompt). Probe on dev, then label pool subsets.
Usage: label_hc.py dev <variant>...  |  label_hc.py pool <variant> <out_name> <n> [offset]"""
import asyncio, json, random, sys
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient

HC = {a: Q[a] for a in ["helpfulness", "correctness"]}
PRICE = {"sort": "price"}
V = {
    "flash_r": dict(model="deepseek/deepseek-v4.1-flash", extra={"reasoning": {"enabled": True, "effort": "medium"}, "provider": PRICE}, mt=4000),
    "flash_rlow": dict(model="deepseek/deepseek-v4.1-flash", extra={"reasoning": {"enabled": True, "effort": "low"}, "provider": PRICE}, mt=3000),
    "flash_nr": dict(model="deepseek/deepseek-v4.1-flash", extra={"provider": PRICE}, mt=None),
    "pro": dict(model="deepseek/deepseek-v4-pro", extra={"provider": PRICE}, mt=None),
    "pro_r": dict(model="deepseek/deepseek-v4-pro", extra={"reasoning": {"enabled": True, "effort": "low"}, "provider": PRICE}, mt=4000),
    "haiku": dict(model="anthropic/claude-haiku-5.5", extra={"provider": {"sort": "price"}}, mt=None),
    "haiku_r": dict(model="anthropic/claude-haiku-5.5", extra={"reasoning": {"enabled": True, "max_tokens": 1024}, "provider": PRICE}, mt=3000),
    "gem31_r": dict(model="google/gemini-3.1-flash-lite", extra={"reasoning": {"enabled": True, "effort": "low"}, "provider": PRICE}, mt=3000),
}

def client(v, seed=None):
    c = V[v]; extra = dict(c["extra"])
    kw = {}
    if seed is not None:
        extra["seed"] = seed; kw["temperature"] = 0.7
    return OpenRouterClient(model=c["model"], concurrency=24, timeout=180, extra_body=extra, **kw)

async def run(v, exs, purpose, seed=None):
    c = client(v, seed)
    out = await label_examples(c, HC, [Example(e.uid, e.state, None, None, e.group, {}) for e in exs], purpose=purpose,
                               max_tokens=V[v]["mt"], progress=False)
    await c.aclose()
    return out, c.stats

def dev_report(tag, out, stats):
    dev = protocol.dev(T); lab = {e.uid: e for e in out}
    vals = {a: np.array([lab[e.uid].labels[a] if lab[e.uid].labels else np.nan for e in dev], float) for a in HC}
    rs = {a: sp(vals[a], [e.gold[a] for e in dev]) for a in HC}
    avg = (vals["helpfulness"] + vals["correctness"]) / 2
    rs_avg = {a: sp(avg, [e.gold[a] for e in dev]) for a in HC}
    nf = sum(e.labels is None for e in out)
    per = stats["cost"] / max(1, stats["calls"])
    print(f"{tag:14s} help {rs['helpfulness']:.3f} corr {rs['correctness']:.3f} | avg->help {rs_avg['helpfulness']:.3f} ->corr {rs_avg['correctness']:.3f} | fail {nf} cost ${stats['cost']:.4f} per-call {per:.2e} out-tok {stats['completion_tokens']/max(1,stats['calls']):.0f}", flush=True)
    return rs

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        for v in sys.argv[2:]:
            out, st = asyncio.run(run(v, protocol.dev(T), f"{T}:r2_dev_{v}"))
            save_jsonl(LD / f"dev_hc_{v}.jsonl", out); dev_report(v, out, st)
    else:
        v, name, n = sys.argv[2], sys.argv[3], int(sys.argv[4]); off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        exs = [e for e in load_jsonl(LD / "silver_v1.jsonl")] + [e for e in load_jsonl(LD / "flash_more.jsonl")]
        random.Random(0).shuffle(exs); exs = exs[off:off + n]
        out, st = asyncio.run(run(v, exs, f"{T}:r2_pool_{v}"))
        save_jsonl(LD / f"{name}.jsonl", [e for e in out if e.labels])
        print("n", len(out), "fail", sum(e.labels is None for e in out), "cost", st["cost"], "per call", st["cost"] / max(1, st["calls"]), flush=True)
