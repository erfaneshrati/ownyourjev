"""2-fold dev estimate of 'seeds + other gold in prompt': label half B with seeds+A as examples and vice versa."""
import asyncio, json
from pathlib import Path
import typer
from benchmarks.v2 import protocol as P
from experiments.banking77 import lab, intents

OUT = Path("data/v2/banking77/devlabels"); OUT.mkdir(parents=True, exist_ok=True)

def main(model: str = "deepseek/deepseek-v4.1-flash", tag: str = "v2", batch: int = 20, soft: bool = False, seed: int = -1, temperature: float = 0.0):
    dev = P.dev("banking77"); sd = P.seeds("banking77")
    A = [i for i in range(200) if i % 2 == 0]; B = [i for i in range(200) if i % 2 == 1]
    cl = lab.make_client(model, temperature=temperature, seed=None if seed < 0 else seed)
    preds = [None] * 200
    async def go():
        for tgt, ex in ((B, A), (A, B)):
            out = await lab.label_texts(cl, [dev[i].state["query"] for i in tgt], purpose=f"banking77:devcv:{tag}", seeds=sd + [dev[i] for i in ex], batch=batch, soft=soft)
            for i, p in zip(tgt, out): preds[i] = p
    asyncio.run(go())
    gold = [e.labels["answer"] for e in dev]
    hard = [(max(p, key=p.get) if isinstance(p, dict) else p) for p in preds]
    acc = sum(h == g for h, g in zip(hard, gold)) / 200
    res = dict(model=model, tag=tag, batch=batch, soft=soft, seed=seed, acc=acc, cost=cl.stats["cost"])
    print(json.dumps(res))
    (OUT / f"cv_{model.split('/')[-1]}_{tag}_b{batch}{'_soft' if soft else ''}{'' if seed<0 else f'_seed{seed}'}.json").write_text(json.dumps({"res": res, "preds": preds}))

if __name__ == "__main__":
    typer.run(main)
