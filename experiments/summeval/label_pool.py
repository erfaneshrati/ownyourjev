"""Label eval_inputs or a synthetic file with an LLM (soft labels). usage: label_pool.py <src: eval|synth_x> <variant> [model]"""
import asyncio, sys
from common import *
from ownyourjev.llm import OpenRouterClient
from ownyourjev.label import label_examples
from ownyourjev.data import read_jsonl, write_jsonl
from guidance import GUIDANCE

async def main(src, variant, model, seed=None):
    exs = P.eval_inputs(T) if src == "eval" else read_jsonl(DATA / f"{src}.jsonl")
    gi = GUIDANCE[variant]
    cl = OpenRouterClient(model=model, concurrency=48, **({'temperature': 0.7, 'extra_body': {'seed': seed}} if seed is not None else {}))
    tag = f'_s{seed}' if seed is not None else ''
    lab = await label_examples(cl, rich_questions() if gi.get("rich") else {d: QS[d] for d in DIMS}, exs,
                               purpose=f"{T}:pool:{src}:{variant}{tag}", guidance=gi.get("text"), distribution=True,
                               max_tokens=gi.get("max_tokens"), progress=False)
    write_jsonl(DATA / f"lab_{src}__{variant}{tag}__{model.split('/')[-1]}.jsonl", lab)
    print(src, variant, model, "ok", sum(e.labels is not None for e in lab), "/", len(lab), f"cost={cl.stats['cost']:.4f}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "qwen/qwen3.7-flash", int(sys.argv[4]) if len(sys.argv) > 4 else None))
