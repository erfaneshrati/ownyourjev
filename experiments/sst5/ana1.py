import sys; sys.path.insert(0,"experiments/sst5")
from lab import *; from calib import *
names = sys.argv[1].split(",")
uids=[e.uid for e in DEV]
ps={n:probs_of(load(n),uids) for n in names}
for n,p in ps.items(): print(n, "raw", round((p.argmax(1)==DEVY).mean(),3), "cv-bias", round(cv_rep(p,DEVY),3))
avg=np.mean(list(ps.values()),0); print("avg raw", (avg.argmax(1)==DEVY).mean(), "cv-bias", round(cv_rep(avg,DEVY),3), "spearman", stats.spearmanr(avg@np.arange(5),DEVY).statistic)
