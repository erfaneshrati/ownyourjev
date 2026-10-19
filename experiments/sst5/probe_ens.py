import sys; sys.path.insert(0,"experiments/sst5")
from lab import *
exs=DEV[100:]; uids=[e.uid for e in exs]; y=np.array([e.labels["answer"] for e in exs])
base=probs_of(load("dev_v1_deepseek-v4.1-flash"),uids).argmax(1)==y
for combo in sys.argv[1:]:
    p=np.mean([probs_of(load(n),uids) for n in combo.split(",")],0); a=p.argmax(1)==y
    rng=np.random.default_rng(0); d=[a[i].mean()-base[i].mean() for i in (rng.integers(0,100,100) for _ in range(5000))]
    print(f"{combo}: B acc {a.mean():.3f} diff {a.mean()-base.mean():+.3f} CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
