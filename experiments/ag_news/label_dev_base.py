"""Baseline labeler variants on dev for all three tasks: plain (v1 prompt) and seeded (v2 seeds), plus soft."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oyj_common import *  # noqa

for t in sys.argv[1:] or ["ag_news", "financial_phrasebank", "emotion"]:
    dev, seeds = protocol.dev(t), protocol.seeds(t)
    report(t, run_label(t, dev, "plain"), dev, "plain (v1 prompt)")
    report(t, run_label(t, dev, "seeded", seeds=seeds), dev, "v2 seeds")
    report(t, run_label(t, dev, "seeded_soft", seeds=seeds, distribution=True), dev, "v2 seeds soft")
