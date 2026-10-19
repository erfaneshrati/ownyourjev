"""PI student. usage: i3_train.py <cfg> <data: pool|pool_synth|...> [--seed s]"""
import argparse, random
from common import *
from ownyourjev.train import TrainConfig
T = "prompt_injections"
CFGS = {
    "gte_e5": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=5, batch_size=16, max_len=256, patience=5),
    "gte_e8": dict(backbone="gte-modernbert", method="full", lr=3e-5, epochs=8, batch_size=16, max_len=256, patience=8),
    "ub_e5": dict(backbone="ultrabert", method="full", lr=1.5e-5, epochs=5, batch_size=16, max_len=256, patience=5),
}
D = ROOT / "data/v2/prompt_injections"
def data(name):
    held = {e.uid for e in protocol.seeds(T)} | {e.uid for e in protocol.dev(T)}
    pool = [e for e in load_jsonl(D / "pool_labels.jsonl") if e.uid not in held]
    syn = [e for e in load_jsonl(D / "synth_labels.jsonl") if e.meta.get("agree")]
    if name == "pool": return pool
    if name == "pool_synth": return pool + syn
    if name == "pool_synth_seeds": return pool + syn + protocol.seeds(T)
    if name == "synth": return syn
    raise KeyError(name)
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tr = data(a.data); random.Random(a.seed).shuffle(tr)
nv = max(40, len(tr) // 10); va, tr = tr[:nv], tr[nv:]
dev = protocol.dev(T); y = np.array([e.labels["answer"] for e in dev])
name = f"{a.cfg}__{a.data}__s{a.seed}"; outd = D / "models" / name
res = train_save(T, tr, va, TrainConfig(**CFGS[a.cfg], seed=a.seed, gpu_need_gb=6.0), outd)
p = torch_probs(res, dev); np.save(outd / "dev_probs.npy", p)
print(f"[{name}] n={len(tr)} best_epoch={res.best_epoch} dev acc {acc_f1(y,p)[0]:.3f} A {acc_f1(y[:100],p[:100])[0]:.3f} B {acc_f1(y[100:],p[100:])[0]:.3f} f1 {acc_f1(y,p)[1]:.3f} best-thr {best_thr(y,p,'acc')}", flush=True)
