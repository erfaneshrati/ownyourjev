"""Combine labelers: flash & qwen on all; gemini-3.1-flash-lite & deepseek-v4-pro where they disagree."""
import json, collections
from pathlib import Path
ORDER = ["deepseek-v4-pro", "deepseek-v4.1-flash", "gemini-3.1-flash-lite", "qwen3.7-flash"]  # tie-break priority

def combine(L):  # L: {labeler: label}
    c = collections.Counter(v for v in L.values() if v)
    top = max(c.values())
    cands = [k for k in c if c[k] == top]
    for o in ORDER:
        if L.get(o) in cands:
            return L[o], {k: n / sum(c.values()) for k, n in c.items()}
    return cands[0], {k: n / sum(c.values()) for k, n in c.items()}

if __name__ == "__main__":
    D = Path("data/v2/banking77/pool")
    f = json.loads((D / "deepseek-v4.1-flash_v2.json").read_text()); q = json.loads((D / "qwen3.7-flash_v2.json").read_text())
    g = json.loads((D / "gemini-3.1-flash-lite_v2_sub.json").read_text()); p = json.loads((D / "deepseek-v4-pro_v2_sub.json").read_text())
    hard, soft = {}, {}
    for u in f:
        if f[u] == q[u]:
            hard[u] = f[u]; soft[u] = {f[u]: 1.0}
        else:
            hard[u], soft[u] = combine({"deepseek-v4.1-flash": f[u], "qwen3.7-flash": q[u], "gemini-3.1-flash-lite": g.get(u), "deepseek-v4-pro": p.get(u)})
    (D / "vote_hard.json").write_text(json.dumps(hard)); (D / "vote_soft.json").write_text(json.dumps(soft))
    print(sum(hard[u] != f[u] for u in f), "changed vs flash")
