"""Label 5000 emotion pool texts (the v1-labeled uids first, topped up at random) with G1 guidance + v2 seeds."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ag_news")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import guidance as G
T = "emotion"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
pool = protocol.pool(T)
v1 = set(v1_silver(T, "silver_seeded"))
first = [e for e in pool if e.uid in v1]
rest = [e for e in pool if e.uid not in v1]
random.Random(0).shuffle(rest)
exs = (first + rest)[:N]
res = run_label(T, exs, f"g1_seeds_{N}", guidance=G.G1, seeds=protocol.seeds(T), split="pool")
print("labeled", sum(e.labels is not None for e in res), "of", len(res))
