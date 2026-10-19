import sys; sys.path.insert(0,"experiments/sst5")
import numpy as np, torch
from lab import *; from calib import fit_bias, apply
from ownyourjev.models.decision import DecisionModel
from ownyourjev.train import predict_probs
from transformers import AutoTokenizer
L=load("pool_v1_deepseek-v4.1-flash_9000")
excl={e.uid for e in DEV}|{e.uid for e in SEEDS}
pool=[e for e in P.pool(T) if e.uid not in excl and e.uid in L and L[e.uid]["labels"]]
idx=np.random.default_rng(0).permutation(len(pool)); val=[pool[i] for i in idx[:400]]
tv=np.array([L[e.uid]["soft"] for e in val]); ty=tv.argmax(1)
for name in sys.argv[1:]:
    d=f"data/v2/sst5/models/{name}"
    m=DecisionModel.load(d, device="cuda:1"); tok=AutoTokenizer.from_pretrained(d+"/backbone")
    pv=predict_probs(m,tok,val); pd=predict_probs(m,tok,DEV)
    b=fit_bias(pv,ty); print(name,"bias",b.round(2),"val agree raw",(pv.argmax(1)==ty).mean(),"->",(apply(pv,b)==ty).mean())
    print(" dev raw",(pd.argmax(1)==DEVY).mean(),"dev w/ silver-bias",(apply(pd,b)==DEVY).mean(), np.bincount(apply(pd,b),minlength=5))
