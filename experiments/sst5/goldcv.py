import sys; sys.path.insert(0,"experiments/sst5")
import numpy as np; from lab import DEVY; from scipy import stats
for pre in sys.argv[1:]:
    p=np.zeros((200,5))
    for k in (0,1):
        d=f"data/v2/sst5/runs/{pre}{k}"; h=np.load(d+"/dev_held.npy"); p[h]=np.load(d+"/dev_probs.npy")[h]
    print(pre,"cv acc",(p.argmax(1)==DEVY).mean(),"spearman",round(stats.spearmanr(p@np.arange(5),DEVY).statistic,3), np.bincount(p.argmax(1),minlength=5))
