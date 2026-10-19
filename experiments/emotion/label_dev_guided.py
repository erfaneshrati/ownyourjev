import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ag_news")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa
import guidance as G
T = "emotion"
dev, seeds = protocol.dev(T), protocol.seeds(T)
name = sys.argv[1]; gname = sys.argv[2]; use_seeds = "--seeds" in sys.argv; soft = "--soft" in sys.argv
model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), None)
report(T, run_label(T, dev, name, guidance=getattr(G, gname), seeds=seeds if use_seeds else None, distribution=soft, model=model), dev, name)
