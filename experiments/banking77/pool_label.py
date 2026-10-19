"""Label the whole pool with a labeler, prompt = intent descriptions + guidelines + all 220 gold examples."""
import asyncio, json
from pathlib import Path
import typer
from benchmarks.v2 import protocol as P
from experiments.banking77 import lab

OUT = Path("data/v2/banking77/pool"); OUT.mkdir(parents=True, exist_ok=True)

def main(model: str = "deepseek/deepseek-v4.1-flash", tag: str = "v2", batch: int = 50, only: str = "", devA_only: bool = False):
    pool = P.pool("banking77"); gold = P.seeds("banking77") + P.dev("banking77")
    ex = pool
    if devA_only:  # clean estimate: prompt sees seeds + dev half A only (dev half B stays unseen)
        d = P.dev("banking77"); gold = P.seeds("banking77") + [d[i] for i in range(0, 200, 2)]; tag += "_devA"
    if only:  # file with list of uids to label
        keep = set(json.loads(Path(only).read_text())); ex = [e for e in pool if e.uid in keep]
    cl = lab.make_client(model)
    preds = asyncio.run(lab.label_texts(cl, [e.state["query"] for e in ex], purpose=f"banking77:pool:{tag}", seeds=gold, batch=batch))
    name = f"{model.split('/')[-1]}_{tag}{'_sub' if only else ''}"
    (OUT / f"{name}.json").write_text(json.dumps({e.uid: p for e, p in zip(ex, preds)}))
    print(json.dumps(dict(model=model, n=len(ex), none=sum(p is None for p in preds), cost=cl.stats["cost"])))

if __name__ == "__main__":
    typer.run(main)
