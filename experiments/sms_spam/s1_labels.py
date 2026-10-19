"""Assemble SMS labels: v1 seeded labels (reused) + label the eval texts v1 never labeled (its own seeds) with the v2 seeds."""
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
T = "sms_spam"
pool = protocol.pool(T)
plain, seeded = load_v1(T, "silver"), load_v1(T, "silver_seeded")
dis = [u for u in seeded if u in plain and plain[u]["answer"] != seeded[u]["answer"]]
print("v1 labeled", len(seeded), "plain/seeded disagree", len(dis), "seeded pos", sum(v["answer"] for v in seeded.values()))
missing = [e for e in pool if e.uid not in seeded]
print("missing", len(missing))
client = OpenRouterClient()
lab = run(label_examples(client, Q(T), missing, purpose="sms_spam:label_missing_v2seeded", seeds=protocol.seeds(T)))
out = []
for e in pool:
    if e.uid in seeded:
        out.append(Example(e.uid, e.state, seeded[e.uid], None, "", {"src": "v1_seeded", "v1_plain": plain.get(e.uid, {}).get("answer")}))
for e in lab:
    if e.labels:
        out.append(Example(e.uid, e.state, e.labels, None, "", {"src": "v2_seeded"}))
print("total labeled", len(out))
save_jsonl(ROOT / "data/v2/sms_spam/labels.jsonl", out)
