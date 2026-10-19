"""Label dev with a labeler/prompt variant; print agreement with dev gold (both heads)."""
import asyncio, sys, json
import numpy as np
from common import *
from prompts import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient

def main(variant, model="deepseek/deepseek-v4.1-flash", use_seeds="1", guidance="GUIDANCE_V1", seed=""):
    dev = protocol.dev(T); sd = protocol.seeds(T)
    eb = {"seed": int(seed)} if seed else None
    if not model.startswith("deepseek") and not model.startswith("qwen"):
        eb = {**(eb or {}), "provider": {"sort": "throughput"}}
    cli = OpenRouterClient(model=model, extra_body=eb, temperature=0.0 if not seed else 0.7)
    g = globals().get(guidance) if guidance != "none" else None
    out = asyncio.run(label_examples(cli, Q, dev, purpose=f"tc:dev:{variant}", seeds=sd if use_seeds == "1" else None,
                                     guidance=g, distribution=True, progress=False))
    save_jsonl(D / f"dev_{variant}.jsonl", out)
    nf = sum(e.labels is None for e in out)
    for h in HEADS:
        y = [e.gold[h] for e in dev]
        p = [soft(o, h) for o in out]
        print(report(y, p, f"{variant}:{h}"), f"cvF1={cv_threshold_f1(y, np.nan_to_num(p)):.3f}", f"fail={nf}")
    print("cost", cli.stats)

if __name__ == "__main__":
    main(*sys.argv[1:])
