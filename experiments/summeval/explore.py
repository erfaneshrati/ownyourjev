import json, numpy as np
from collections import Counter
from benchmarks.v2 import protocol as P
from ownyourjev.spec import render_state
T="summeval"
s,d,ev=P.seeds(T),P.dev(T),P.eval_inputs(T)
print(len(s),len(d),len(ev), len({e.group for e in s}),len({e.group for e in d}),len({e.group for e in ev}))
print(Counter(e.group for e in s+d))
for q in ["coherence","consistency","fluency","relevance"]:
    print(q, Counter(e.labels[q] for e in d))
folds=P.eval_folds(T); print([ (len(a),len(b)) for a,b in folds])
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Alibaba-NLP/gte-modernbert-base")
L=[len(tok(render_state(e.state))["input_ids"]) for e in d]
Ls=[len(tok(e.state["summary"])["input_ids"]) for e in d]
print("full tok pct", np.percentile(L,[5,50,95,100]), "summ", np.percentile(Ls,[5,50,95,100]))
print(render_state(d[0].state)[:3000]); print(d[0].labels)
print(P.task_spec(T).parsed["coherence"].describe())
import itertools
sil=[json.loads(l) for l in open("data/silver/summeval/silver_seeded.jsonl")]
print(len(sil), sil[0].keys(), sil[0].get("labels"), sil[0].get("meta",{}).keys() if sil[0].get("meta") else None)
