"""Round 3: more re-pairings of pool sentences. All top-K (K=12) gte-modernbert neighbours of every unique pool sentence
(excluding original pool pairs, dev/seed sentences and pairs already used in rounds 2), scored by the round-2 student,
stratified-sampled by predicted level with quotas. Writes data/v2/stsb/new_repair3.jsonl."""
import json, random, sys, torch
from common import *
from transformers import AutoTokenizer, AutoModel
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
quota = [int(x) for x in sys.argv[1].split(",")]
K = 12
pool = protocol.pool(T)
sents = sorted({s.strip() for e in pool for s in (e.state["sentence_1"], e.state["sentence_2"])})
devs = {s.strip() for e in protocol.dev(T) + protocol.seeds(T) for s in e.state.values()}
sents = [s for s in sents if s not in devs]
key = lambda a, b: tuple(sorted((a, b)))
used = {key(e.state["sentence_1"].strip(), e.state["sentence_2"].strip()) for e in pool}
for f in ["new_repair", "new_repair2"]:
    used |= {key(r["state"]["sentence_1"], r["state"]["sentence_2"]) for r in map(json.loads, open(D / f"{f}.jsonl"))}
dev_ = "cuda:0"
tok = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-modernbert-base"); m = AutoModel.from_pretrained("Alibaba-NLP/gte-modernbert-base").to(dev_).eval()
E = []
with torch.no_grad():
    for i in range(0, len(sents), 128):
        enc = tok(sents[i:i+128], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            E.append(torch.nn.functional.normalize(m(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu())
E = torch.cat(E); S = E @ E.T; S.fill_diagonal_(-1)
top = S.topk(K, dim=1).indices.numpy()
cand = set()
for i in range(len(sents)):
    for j in top[i]:
        k = key(sents[i], sents[j])
        if k not in used: cand.add(k)
rng = random.Random(3); cand = sorted(cand); rng.shuffle(cand)
cand = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in cand]
print("candidates", len(cand), flush=True)
del m; torch.cuda.empty_cache()
st = DecisionModel.load(D / "models/final_r2", device=dev_); stok = AutoTokenizer.from_pretrained(D / "models/final_r2/backbone")
exs = [Example(f"rp3-{k}", {"sentence_1": a, "sentence_2": b}, None) for k, (a, b) in enumerate(cand)]
p = predict_probs(st, stok, exs, device=dev_, batch_size=256) @ np.arange(6)
print("pred hist", np.histogram(p, bins=[0,1,2,3,4,5.01])[0])
bins = np.minimum(p.astype(int), 4); out = []
for b in range(5):
    idx = [i for i in range(len(exs)) if bins[i] == b]; rng.shuffle(idx); out += idx[:quota[b]]
rng.shuffle(out)
print("selected", len(out), np.histogram(p[out], bins=[0,1,2,3,4,5.01])[0])
with open(D / "new_repair3.jsonl", "w") as f:
    for i in out:
        f.write(json.dumps({"uid": exs[i].uid, "state": exs[i].state, "pred": float(p[i])}) + "\n")
