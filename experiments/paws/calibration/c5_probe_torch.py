"""Probe torch checkpoints against the frozen test probs (first n test inputs, no labels)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common import *
from transformers import AutoTokenizer
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
ev = protocol.eval_inputs("paws"); frozen = np.load(ROOT / "data/v2/paws/eval_probs.npy")
n = 256
for name in sys.argv[1:]:
    d = ROOT / "data/v2/paws/models" / name
    m = DecisionModel.load(d, device="cpu"); tok = AutoTokenizer.from_pretrained(d / "backbone")
    p = predict_probs(m, tok, ev[:n]).reshape(-1)
    print(name, "torch max|diff|", float(np.abs(p - frozen[:n]).max()), "mean", float(np.abs(p - frozen[:n]).mean()))
