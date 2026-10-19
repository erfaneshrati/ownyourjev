"""Recompute a finished task's secondary heads (silver variant) with the current, cross-fitted procedure."""

import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.run import secondary_head_probs
from benchmarks.tasks import get_task
from ownyourjev.data import kfold, read_jsonl
from ownyourjev.evaluate import evaluate
from ownyourjev.train import TrainConfig

name = sys.argv[1]
task = get_task(name)
spec = task.spec
res_path = Path("benchmarks/results") / name / "result.json"
r = json.loads(res_path.read_text())
cfg = TrainConfig(**r["variants"]["silver"]["recipe"])
evalset = task.eval()
folds = kfold(evalset, spec.cross_fit_folds, seed=0, grouped=spec.grouped) if spec.cross_fit_folds else None
groups = [e.group for e in evalset] if spec.grouped else None
data = [e for e in read_jsonl(Path("data/silver") / name / "silver.jsonl") if e.valid]
vdir = res_path.parent / "silver"
for h, qh in spec.parsed.items():
    if h == spec.primary_head:
        continue
    p, gpu = secondary_head_probs(qh, h, [e for e in data if h in e.labels], evalset, folds, cfg)
    m = evaluate(qh, [e.gold[h] for e in evalset], p, groups)
    r["variants"]["silver"]["other_heads"][h] = {**m, "gpu_s": gpu, "cross_fit": bool(folds)}
    np.save(vdir / f"eval_probs_{h}.npy", p)
    print(h, {k: round(v, 4) for k, v in m.items() if k in ("group_spearman", "spearman", "f1", "accuracy")}, flush=True)
res_path.write_text(json.dumps(r, indent=1, default=float))
