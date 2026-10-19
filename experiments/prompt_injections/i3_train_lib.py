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
