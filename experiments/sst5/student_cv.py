import sys; sys.path.insert(0,"experiments/sst5")
import numpy as np; from lab import DEVY; from calib import cv_rep; from scipy import stats
for n in sys.argv[1:]:
    ps=[np.load(f"data/v2/sst5/runs/{x}/dev_probs.npy") for x in n.split("+")]; p=np.mean(ps,0)
    print(n, "raw", round((p.argmax(1)==DEVY).mean(),3), "cv-bias", round(cv_rep(p,DEVY),3), "spearman", round(stats.spearmanr(p@np.arange(5),DEVY).statistic,3), "dist", np.bincount(p.argmax(1),minlength=5))
