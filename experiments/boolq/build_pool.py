"""Combine pool labels: qwen3.7-flash (all) + v1 deepseek-seeded (4.9k) + gemini-3.1-flash-lite tie-breaks."""
from common import *
q = {e.uid: e for e in read_jsonl(DATA / 'pool_qwen_dist.jsonl')}
v1 = {e.uid: e for e in read_jsonl(DATA / 'v1_seeded.jsonl')}
gm = {e.uid: e for e in read_jsonl(DATA / 'pool_gemini_tb.jsonl') if e.labels}
out, src = [], {}
for u, e in q.items():
    if u in gm:
        lab, s = gm[u].labels[QID], "gemini_tb"
    elif u in v1:
        lab, s = e.labels[QID], "qwen+v1"
    else:
        lab, s = e.labels[QID], "qwen"
    src[s] = src.get(s, 0) + 1
    out.append(Example(u, e.state, {QID: lab}, None, "", {"src": s}))
write_jsonl(DATA / 'pool_comb.jsonl', out)
print(src, "yes_rate", np.mean([e.labels[QID] for e in out]))
