"""Uniform weight averaging ("model soup") of several MultiHead runs fine-tuned from the same init.
Usage: soup.py OUT_NAME IN1 IN2 ..."""
import sys, shutil
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from mt_model import MultiHead
out, ins = sys.argv[1], sys.argv[2:]
ms = [MultiHead.load(DATA / "models" / n)[0] for n in ins]
m, tok = MultiHead.load(DATA / "models" / ins[0])
sd = {k: (sum(x.state_dict()[k].float() for x in ms) / len(ms)).to(v.dtype) if v.is_floating_point() else v for k, v in m.state_dict().items()}
m.load_state_dict(sd)
m.temperature.fill_(1.0)
import json
q = json.loads((DATA / "models" / ins[0] / "meta.json").read_text())["question"]
m.save(DATA / "models" / out, tok, q)
print("saved soup", out)
