from benchmarks.v2 import protocol as P
d=P.dev("helpsteer2")[:100]
A=["helpfulness","correctness","coherence","complexity","verbosity"]
low=[x for x in d if x.gold["helpfulness"]<=1][:6]
for x in low:
    print("=====",[x.gold[a] for a in A]); print("PROMPT:",x.state["prompt"][-700:]); print("RESP:",x.state["response"][:900])
