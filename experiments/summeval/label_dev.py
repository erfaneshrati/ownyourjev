"""Label dev (or seeds) with an LLM and report per-dim group Spearman vs gold.
usage: label_dev.py <model> <variant> [--seeds]"""
import asyncio, sys, json
import numpy as np
from common import *
from ownyourjev.llm import OpenRouterClient
from ownyourjev.label import label_examples
from ownyourjev.data import write_jsonl
from guidance import GUIDANCE

async def main(model, variant, extra=None, tag=''):
    exs = P.dev(T)
    gi = GUIDANCE[variant]
    qs = rich_questions() if gi.get("rich") else {d: QS[d] for d in DIMS}
    kw = dict(extra_body=extra) if extra else {}
    cl = OpenRouterClient(model=model, concurrency=32, temperature=0.7 if extra else 0.0, **kw)
    seeds = P.seeds(T) if gi.get("seeds") else None
    lab = await label_examples(cl, qs, exs, purpose=f"{T}:dev:{variant}{tag}:{model.split('/')[-1]}", guidance=gi.get("text"),
                               distribution=gi.get("dist", True), seeds=seeds, system_preamble=gi.get("preamble"), max_tokens=gi.get("max_tokens"))
    ok = [e for e in lab if e.labels]
    pred = {}
    for d in DIMS:
        if gi.get("dist", True):
            pred[d] = np.array([np.dot(e.meta["soft"][d], np.arange(5)) if e.labels else 2.0 for e in lab])
        else:
            pred[d] = np.array([e.labels[d] if e.labels else 2.0 for e in lab])
    res = score_dev(pred, exs)
    # also with gold labels from protocol for comparison
    print(model, variant, f"ok={len(ok)}/{len(lab)}", {k: round(v, 3) for k, v in res.items()}, f"cost={cl.stats['cost']:.4f}")
    out = DATA / "dev_labels"; out.mkdir(exist_ok=True)
    write_jsonl(out / f"{variant}{tag}__{model.replace('/', '_')}.jsonl", lab)
    with open(RES / "dev_labeler_log.jsonl", "a") as f:
        f.write(json.dumps({"model": model, "variant": variant + tag, **res, "cost": cl.stats["cost"]}) + "\n")

if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    asyncio.run(main(sys.argv[1], sys.argv[2], {"seed": seed} if seed is not None else None, f"_s{seed}" if seed is not None else ""))
