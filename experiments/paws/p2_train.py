"""PAWS student training on LLM labels; dev = gold. usage: p2_train.py <cfg> <labelset> [--seed s]"""
import argparse, random
from common import *
from ownyourjev.train import TrainConfig
T = "paws"
CFGS = {
    "gte_e3": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=3, batch_size=32, max_len=128),
    "gte_e5": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=5, batch_size=32, max_len=128, patience=5),
    "ub_e3": dict(backbone="ultrabert", method="full", lr=1.15e-5, epochs=3, batch_size=32, max_len=128),
    "ub_e5": dict(backbone="ultrabert", method="full", lr=1.15e-5, epochs=5, batch_size=32, max_len=128, patience=2),
    "ub_e2_soft": dict(backbone="ultrabert", method="full", lr=1.15e-5, epochs=2, batch_size=32, max_len=128, soft_labels=True),
    "ub_e3_soft": dict(backbone="ultrabert", method="full", lr=1.15e-5, epochs=3, batch_size=32, max_len=128, soft_labels=True),
    "gte_e3_soft": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=3, batch_size=32, max_len=128, soft_labels=True),
    "gte_e8_lr2": dict(backbone="gte-modernbert", method="full", lr=2e-5, epochs=8, batch_size=32, max_len=128, patience=8),
}
def labelset(name):
    held = {e.uid for e in protocol.seeds(T)} | {e.uid for e in protocol.dev(T)}
    pool = {e.uid: e for e in protocol.pool(T)}
    if name in ("v1_seeded", "v1_plain", "v1_agree"):
        s, p = load_v1(T, "silver_seeded"), load_v1(T, "silver")
        if name == "v1_seeded": d = s
        elif name == "v1_plain": d = p
        else: d = {u: v for u, v in s.items() if u in p and p[u]["answer"] == v["answer"]}
        return [Example(u, pool[u].state, v, None) for u, v in d.items()]
    if name == "v1s+v2more":
        return labelset("v1_seeded") + [e for e in load_jsonl(ROOT / "data/v2/paws/v2_seeded_more.jsonl") if e.uid not in held]
    if name.startswith("all+pseudo:"):  # all LLM labels + teacher pseudo labels for the rest
        llm = labelset("v1s+v2more"); u = {e.uid for e in llm}
        return llm + [e for e in load_jsonl(ROOT / f"data/v2/paws/pseudo_{name.split(':')[1]}.jsonl") if e.uid not in held and e.uid not in u]
    if name.startswith("v1s+pseudo:"):  # v1 seeded LLM labels + teacher pseudo labels for the rest of the pool
        return labelset("v1_seeded") + [e for e in load_jsonl(ROOT / f"data/v2/paws/pseudo_{name.split(':')[1]}.jsonl") if e.uid not in held]
    return [e for e in load_jsonl(ROOT / f"data/v2/paws/{name}.jsonl") if e.labels and e.uid not in held]
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("labels"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tr = labelset(a.labels); random.Random(0).shuffle(tr)
nv = min(500, len(tr) // 10); va, tr = tr[:nv], tr[nv:]
print("train", len(tr), "pos", sum(e.labels["answer"] for e in tr), flush=True)
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
name = f"{a.cfg}__{a.labels.replace(':','-')}__s{a.seed}"
outd = ROOT / "data/v2/paws/models" / name
res = train_save(T, tr, va, TrainConfig(**CFGS[a.cfg], seed=a.seed, gpu_need_gb=6.0), outd)
p = torch_probs(res, dev); np.save(outd / "dev_probs.npy", p)
print(f"[{name}] best_epoch={res.best_epoch} hist={[round(h['val_loss'],3) for h in res.history]} dev acc {acc_f1(y,p)[0]:.3f} A {acc_f1(y[:100],p[:100])[0]:.3f} B {acc_f1(y[100:],p[100:])[0]:.3f}", flush=True)
