"""Sentence-level fact verification of each summary against its article (LLM), -> per-summary support scores.
usage: verify.py <src: dev|eval|synth_x> [seed]"""
import asyncio, re, sys, json
import numpy as np
from common import *
from ownyourjev.llm import OpenRouterClient
from ownyourjev.data import read_jsonl, write_jsonl, Example
from dataclasses import replace

SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")
VAL = {"supported": 1.0, "minor": 0.6, "unsupported": 0.0, "contradicted": 0.0}

def sents(summary):
    s = [x.strip() for x in SPLIT.split(summary.strip()) if len(x.strip()) > 3]
    return s[:10] or [summary.strip()]

SYS = ("You are a careful fact-checker. Given a news ARTICLE and numbered SUMMARY SENTENCES, judge each sentence ONLY against the article. "
       "Verdicts: \"supported\" = every claim is stated in or directly entailed by the article (verbatim copies and faithful paraphrases are supported; "
       "lower-casing and tokenisation like \"it 's\" or \"-lrb-\" are not errors; a sentence cut off mid-way is judged on what it says); "
       "\"minor\" = essentially faithful but with a small imprecision or an unsupported detail of little importance; "
       "\"unsupported\" = contains a claim not in the article (invented fact, wrong number/name/date, wrong attribution of a quote or action, wrong outcome); "
       "\"contradicted\" = a claim the article contradicts. Reply with JSON only.")

def msgs(state):
    ss = sents(state["summary"])
    user = "ARTICLE:\n" + state["source_document"].strip() + "\n\nSUMMARY SENTENCES:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(ss))
    user += '\n\nReturn {"verdicts": ["<verdict for 1>", "<verdict for 2>", ...]} with exactly ' + str(len(ss)) + " verdicts."
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}], len(ss)

def score(v):
    x = [VAL.get(str(t).strip().lower(), np.nan) for t in v]
    x = [t for t in x if t == t]
    if not x: return None
    return {"v_mean": float(np.mean(x)), "v_min": float(np.min(x)), "v_n_bad": float(sum(t < 0.5 for t in x)), "v_n": len(x)}

async def main(src, seed):
    exs = P.dev(T) if src == "dev" else P.eval_inputs(T) if src == "eval" else read_jsonl(DATA / f"{src}.jsonl")
    kw = {"temperature": 0.7, "extra_body": {"seed": seed}} if seed is not None else {}
    cl = OpenRouterClient(model="qwen/qwen3.7-flash", concurrency=48, **kw)
    async def one(e):
        m, n = msgs(e.state)
        for _ in range(2):
            try:
                raw = await cl.complete_json(m, purpose=f"{T}:verify:{src}", max_tokens=40 + 12 * n)
                v = raw.get("verdicts") if isinstance(raw, dict) else None
                if isinstance(v, list) and len(v) == n:
                    s = score(v)
                    if s: return replace(e, meta={**e.meta, "verify": s, "verdicts": v})
            except Exception:
                pass
        return replace(e, meta={**e.meta, "verify": None})
    out = await cl.map(exs, one, desc="verify", progress=False)
    tag = f"_s{seed}" if seed is not None else ""
    write_jsonl(DATA / f"verify_{src.replace('/', '_')}{tag}.jsonl", out)
    print(src, "ok", sum(e.meta["verify"] is not None for e in out), "/", len(out), f"cost={cl.stats['cost']:.4f}")
    if src == "dev":
        g = [e.group for e in out]
        for k in ("v_mean", "v_min", "v_n_bad"):
            x = np.array([e.meta["verify"][k] if e.meta["verify"] else np.nan for e in out]); x[np.isnan(x)] = np.nanmean(x)
            print(k, {d: round(group_spearman([e.labels[d] for e in out], x * (-1 if k == "v_n_bad" else 1), g), 3) for d in DIMS})

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))
