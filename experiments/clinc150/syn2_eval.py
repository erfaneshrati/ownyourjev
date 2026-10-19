"""oos recall of existing models on the held-out round-2 near-miss synthetic oos (computed with a torch model)."""
import sys, random
import numpy as np
from ownyourjev.data import read_jsonl
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from transformers import AutoTokenizer
from experiments.clinc150.common import *
syn2 = read_jsonl("data/v2/clinc150/synth_oos2.jsonl"); random.Random(1).shuffle(syn2); va = syn2[:250]
for n in sys.argv[1:]:
    d = f"data/v2/clinc150/models/{n}"
    m = DecisionModel.load(d, device="cpu"); tok = AutoTokenizer.from_pretrained(d + "/backbone")
    p = predict_probs(m, tok, va); np.save(d + "/syn2va_probs.npy", p)
    print(n, " ".join(f"t{t}:{(apply_threshold(p, t).argmax(1) == OOS).mean():.3f}" for t in [0, 0.2, 0.3, 0.4, 0.5]))
