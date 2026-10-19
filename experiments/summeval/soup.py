"""Uniform weight soup of several saved multi-head models (same config, different seeds); eval on dev.
usage: soup.py <out_dir> <model_dir1> <model_dir2> ..."""
import sys, shutil
from pathlib import Path
import numpy as np, torch
from common import *
import mh
out, ins = Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]]
sds = [torch.load(p / "mh.pt", map_location="cpu") for p in ins]
avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
out.mkdir(parents=True, exist_ok=True)
for f in ("mh_cfg.json",):
    shutil.copy(ins[0] / f, out / f)
shutil.copytree(ins[0] / "backbone", out / "backbone", dirs_exist_ok=True)
torch.save(avg, out / "mh.pt")
dev = P.dev(T)
dev_feats = mh.featurize(dev)
for p in ins + [out]:
    m, tok = mh.load(p, "cuda:0" if torch.cuda.is_available() else "cpu")
    pr = mh.predict(m, tok, dev, m.temp.device, feats=dev_feats)
    r = score_dev({d: pr[:, j] @ np.arange(5) for j, d in enumerate(DIMS)}, dev)
    print(p.name, {k: round(v, 4) for k, v in r.items()})
