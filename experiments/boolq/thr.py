import sys, numpy as np
from common import *
g = np.array([e.gold[QID] for e in protocol.dev(TASK)])
for f in sys.argv[1:]:
    p = np.load(f)
    row = [f"{t:.2f}:{np.mean((p>=t)==g):.3f}" for t in [0.3, 0.4, 0.5, 0.6, 0.7]]
    A, B = slice(0, 100), slice(100, 200)
    ts = np.linspace(0.05, 0.95, 91)
    bA = ts[np.argmax([np.mean((p[A]>=t)==g[A]) for t in ts])]; bB = ts[np.argmax([np.mean((p[B]>=t)==g[B]) for t in ts])]
    print(Path(f).parent.name, " ".join(row), f"| tA={bA:.2f}->B {np.mean((p[B]>=bA)==g[B]):.3f} (B@.5 {np.mean((p[B]>=.5)==g[B]):.3f}) tB={bB:.2f}->A {np.mean((p[A]>=bB)==g[A]):.3f} (A@.5 {np.mean((p[A]>=.5)==g[A]):.3f})")
