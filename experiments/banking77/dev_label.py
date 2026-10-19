"""Label dev with a labeler config; report accuracy on dev half A / half B / all and save predictions."""
import asyncio, json, sys, collections
from pathlib import Path
import typer
from benchmarks.v2 import protocol as P
from experiments.banking77 import lab

OUT = Path("data/v2/banking77/devlabels"); OUT.mkdir(parents=True, exist_ok=True)

def halves(dev):
    return [i for i in range(len(dev)) if i % 2 == 0], [i for i in range(len(dev)) if i % 2 == 1]

def main(model: str = "deepseek/deepseek-v4.1-flash", tag: str = "v1", batch: int = 20, seeds: bool = True, soft: bool = False, seed: int = -1, temperature: float = 0.0, errors: bool = False, devA: bool = False):
    dev = P.dev("banking77"); sd = P.seeds("banking77") if seeds else None
    A0, B0 = halves(dev)
    if devA:
        sd = (sd or []) + [dev[i] for i in A0]
    texts = [e.state["query"] for e in dev]; gold = [e.labels["answer"] for e in dev]
    cl = lab.make_client(model, temperature=temperature, seed=None if seed < 0 else seed)
    preds = asyncio.run(lab.label_texts(cl, texts, purpose=f"banking77:dev:{tag}", seeds=sd, batch=batch, soft=soft))
    hard = [(max(p, key=p.get) if isinstance(p, dict) else p) for p in preds]
    A, B = halves(dev)
    acc = lambda ix: sum(hard[i] == gold[i] for i in ix) / len(ix)
    res = dict(model=model, tag=tag, batch=batch, seeds=seeds, soft=soft, accA=acc(A), accB=acc(B), acc=acc(range(len(dev))), n_none=sum(h is None for h in hard), cost=cl.stats["cost"])
    print(json.dumps(res))
    name = f"{model.split('/')[-1]}_{tag}_b{batch}{'_s' if seeds else ''}{'_soft' if soft else ''}{'' if seed<0 else f'_seed{seed}'}{'_devA' if devA else ''}"
    (OUT / f"{name}.json").write_text(json.dumps({"res": res, "preds": preds}))
    if errors:
        for i in A:
            if hard[i] != gold[i]:
                print(f"GOLD {gold[i]:40s} PRED {str(hard[i]):40s} | {texts[i]}")

if __name__ == "__main__":
    typer.run(main)
