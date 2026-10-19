import sys
from common import *
from threshold import boot_threshold, cv_boot_threshold_f1
dev = protocol.dev(T)
for n in sys.argv[1:]:
    p = np.load(D / 'runs' / n / 'dev_probs.npy')
    for i, h in enumerate(HEADS):
        y = [e.gold[h] for e in dev]; t, s, _ = boot_threshold(y, p[:, i])
        print(n, h, 'boot_t', t, 'smoothF1', round(s, 3), 'F1@t', round(f1_at(y, p[:, i], t), 3), 'cv_boot', round(cv_boot_threshold_f1(y, p[:, i], reps=4), 3), flush=True)
