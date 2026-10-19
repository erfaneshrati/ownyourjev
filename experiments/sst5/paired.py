import sys; sys.path.insert(0,"experiments/sst5")
import numpy as np; from lab import DEVY
a=np.load(sys.argv[1]).argmax(1)==DEVY; b=np.load(sys.argv[2]).argmax(1)==DEVY
rng=np.random.default_rng(0); d=[(a[i].mean()-b[i].mean()) for i in (rng.integers(0,200,200) for _ in range(10000))]
print(f"new {a.mean():.3f} old {b.mean():.3f} diff {a.mean()-b.mean():+.3f} 95%CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] P(diff<=0)={np.mean(np.array(d)<=0):.2f}")
