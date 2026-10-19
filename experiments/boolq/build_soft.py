"""Round 2 (a): soft labels = mean vote of the available labelers per pool text
(qwen3.7-flash on all; v1 deepseek-seeded on 4.9k or deepseek-flash plain on the other 4.35k; gemini-3.1-flash-lite on 714 disputed)."""
from common import *
q = {e.uid: e for e in read_jsonl(DATA / 'pool_qwen_dist.jsonl')}
v1 = {e.uid: e for e in read_jsonl(DATA / 'v1_seeded.jsonl')}
ds = {e.uid: e for e in read_jsonl(DATA / 'pool_dsflash.jsonl') if e.labels}
gm = {e.uid: e for e in read_jsonl(DATA / 'pool_gemini_tb.jsonl') if e.labels}
out = []
nv = []
for u, e in q.items():
    votes = [float(e.meta['soft'][QID])]
    for src in (v1, ds, gm):
        if u in src:
            votes.append(float(src[u].labels[QID]))
    p = float(np.mean(votes)); nv.append(len(votes))
    out.append(Example(u, e.state, {QID: p >= 0.5}, None, "", {"soft": {QID: p}, "n_votes": len(votes)}))
write_jsonl(DATA / 'pool_soft.jsonl', out)
ps = np.array([e.meta['soft'][QID] for e in out])
print("n", len(out), "votes", np.bincount(nv), "soft in (0.2,0.8):", int(((ps > .2) & (ps < .8)).sum()), "yes", (ps >= .5).mean())
