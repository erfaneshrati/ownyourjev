import sys; sys.path.insert(0,"experiments/sst5")
from lab import *
uids=[e.uid for e in DEV]
for combo in sys.argv[1:]:
    p=np.mean([probs_of(load(n),uids) for n in combo.split(",")],0); a=p.argmax(1)==DEVY
    print(combo, "acc %.3f A %.3f B %.3f"%(a.mean(),a[:100].mean(),a[100:].mean()))
