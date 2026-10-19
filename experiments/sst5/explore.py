import json, collections
from benchmarks.v2 import protocol as P
s,d,p,e=P.seeds("sst5"),P.dev("sst5"),P.pool("sst5"),P.eval_inputs("sst5")
print(len(s),len(d),len(p),len(e))
print("seed dist",collections.Counter(x.labels["answer"] for x in s))
print("dev dist",sorted(collections.Counter(x.labels["answer"] for x in d).items()))
for x in s: print(x.labels["answer"], x.state["sentence"])
dev_ids={x.uid:x.labels["answer"] for x in d}
for f in ["silver","silver_seeded"]:
    rows=[json.loads(l) for l in open(f"data/silver/sst5/{f}.jsonl")]
    m=[(r["labels"]["answer"],dev_ids[r["uid"]]) for r in rows if r.get("labels") and r["uid"] in dev_ids]
    print(f,len(rows),"dev overlap",len(m),"acc",sum(a==b for a,b in m)/max(1,len(m)))
