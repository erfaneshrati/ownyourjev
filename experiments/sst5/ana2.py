import sys; sys.path.insert(0,"experiments/sst5")
from lab import *
a=load("dev_v1_deepseek-v4.1-flash"); b=load("v1_silver_seeded"); c=load("v1_silver")
uids=[e.uid for e in DEV if e.uid in b]; y=np.array([e.labels["answer"] for e in DEV if e.uid in b])
pa=probs_of(a,uids); pb=probs_of(b,uids); pc=probs_of(c,uids)
for w in [0,0.2,0.35,0.5]:
    p=(1-w)*pa+w*pb; print(w, (p.argmax(1)==y).mean())
print("pc", (pc.argmax(1)==y).mean(), "pb",(pb.argmax(1)==y).mean(), len(y))
