import json, collections, numpy as np
from benchmarks.v2 import protocol as P
from transformers import AutoTokenizer
from scipy.stats import spearmanr
T="helpsteer2"
s,d,p,e=P.seeds(T),P.dev(T),P.pool(T),P.eval_inputs(T)
print(len(s),len(d),len(p),len(e))
print(d[0].uid, d[0].state.keys(), d[0].gold)
A=["helpfulness","correctness","coherence","complexity","verbosity"]
G=np.array([[x.gold[a] for a in A] for x in d])
for i,a in enumerate(A): print(a, collections.Counter(G[:,i]))
print(np.round(np.corrcoef(G.T),2))
tok=AutoTokenizer.from_pretrained("Alibaba-NLP/gte-modernbert-base")
lp=[len(tok(x.state["prompt"])["input_ids"]) for x in d]; lr=[len(tok(x.state["response"])["input_ids"]) for x in d]
for nm,l in [("prompt",lp),("resp",lr),("tot",np.add(lp,lr))]: print(nm, np.percentile(l,[50,75,90,95,99]).round())
# length correlation with gold
for i,a in enumerate(A): print(a, "len_resp rho", round(spearmanr(lr,G[:,i])[0],3))
# pairs in pool
c=collections.Counter(x.state["prompt"] for x in p); print("pool prompts", len(c), collections.Counter(c.values()))
ce=collections.Counter(x.state["prompt"] for x in e); print("eval prompts", len(ce))
dp=set(x.state["prompt"] for x in d); print("dev prompts in pool", sum(1 for x in p if x.state["prompt"] in dp))
sil=[json.loads(l) for l in open("data/silver/helpsteer2/silver.jsonl")]
print(len(sil), sil[0].keys(), sil[0].get("labels"), sil[0].get("meta"))
devu=set(x.uid for x in d)|set(x.uid for x in s)
print("silver in dev/seeds", sum(r["uid"] in devu for r in sil))
