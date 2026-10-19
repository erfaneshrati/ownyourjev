"""Round 2 data: new pairings of pool sentences. Embed unique pool sentences with base gte-modernbert (CLS, normalized),
for each sentence take a random neighbour among its top-30 (excluding its original partner), score candidates with the
round-1 student (F_s0) and sample to a roughly uniform 0-5 predicted-score distribution. Writes data/v2/stsb/new_repair.jsonl."""
import json, random, sys, torch
from common import *
from transformers import AutoTokenizer, AutoModel
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
pool = protocol.pool(T)
excl = {e.uid for e in protocol.dev(T)} | {e.uid for e in protocol.seeds(T)}
sents = sorted({s.strip() for e in pool for s in (e.state["sentence_1"], e.state["sentence_2"])})
orig = {(e.state["sentence_1"].strip(), e.state["sentence_2"].strip()) for e in pool}
orig |= {(b, a) for a, b in orig}
devs = {s.strip() for e in protocol.dev(T) + protocol.seeds(T) for s in e.state.values()}
sents = [s for s in sents if s not in devs]
print("unique sentences", len(sents))
dev_ = "cuda:0"
tok = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-modernbert-base"); m = AutoModel.from_pretrained("Alibaba-NLP/gte-modernbert-base").to(dev_).eval()
E = []
with torch.no_grad():
    for i in range(0, len(sents), 128):
        enc = tok(sents[i:i+128], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = m(**enc).last_hidden_state[:, 0]
        E.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
E = torch.cat(E); S = E @ E.T; S.fill_diagonal_(-1)
top = S.topk(30, dim=1).indices.numpy()
rng = random.Random(0); cand = set()
for i in range(len(sents)):
    for j in rng.sample(list(top[i]), 2):
        a, b = sents[i], sents[j]
        if (a, b) in orig or (b, a) in cand: continue
        cand.add((a, b))
cand = sorted(cand); rng.shuffle(cand)
print("candidates", len(cand))
del m; torch.cuda.empty_cache()
st = DecisionModel.load(D / "models/F_s0", device=dev_); stok = AutoTokenizer.from_pretrained(D / "models/F_s0/backbone")
exs = [Example(f"rp-{k}", {"sentence_1": a, "sentence_2": b}, None) for k, (a, b) in enumerate(cand)]
p = predict_probs(st, stok, exs, device=dev_) @ np.arange(6)
print("pred hist", np.histogram(p, bins=[0,1,2,3,4,5.01])[0])
# stratified sampling: equal per predicted bin
bins = np.minimum(p.astype(int), 4); out = []
OUTF = sys.argv[2] if len(sys.argv) > 2 else "new_repair"
quota = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [N // 5] * 5
taken = set()
if OUTF != "new_repair":
    taken = {json.dumps(r["state"], sort_keys=True) for r in map(json.loads, open(D / "new_repair.jsonl"))}
for b in range(5):
    idx = [i for i in range(len(exs)) if bins[i] == b]; rng.shuffle(idx)
    idx = [i for i in idx if json.dumps(exs[i].state, sort_keys=True) not in taken]
    out += idx[:quota[b]]
print("selected", len(out), np.histogram(p[out], bins=[0,1,2,3,4,5.01])[0])
with open(D / f"{OUTF}.jsonl", "w") as f:
    for i in out:
        f.write(json.dumps({"uid": ("rp2-" if OUTF != "new_repair" else "") + exs[i].uid, "state": exs[i].state, "pred": float(p[i])}) + "\n")
