import sys; sys.path.insert(0,"experiments/sst5")
from lab import *
lab=load(sys.argv[1])
for e in DEV[:100]:
    r=lab[e.uid]; p=r["labels"]["answer"] if r["labels"] else None
    if p!=e.labels["answer"]: print(e.labels["answer"], p, [round(x,2) for x in r["soft"]] if r["soft"] else "", e.state["sentence"])
