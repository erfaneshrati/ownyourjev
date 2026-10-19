import sys; sys.path.insert(0, "experiments/summeval")
from common import *; from feats import *
d = P.dev(T); X = np.stack([features(e.state["summary"], e.state["source_document"]) for e in d]); g=[e.group for e in d]
for j, n in enumerate(FEAT_NAMES):
    print(f"{n:14s}", " ".join(f"{group_spearman([e.labels[k] for e in d], X[:,j], g):+.2f}" for k in DIMS))
