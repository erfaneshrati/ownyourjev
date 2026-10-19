"""Cheap lexical alignment features between summary and the FULL article (computed in the request path)."""
import re
import numpy as np

_tok = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def toks(s):
    s = s.lower().replace("-lrb-", "(").replace("-rrb-", ")").replace("``", '"').replace("''", '"')
    s = re.sub(r" '(s|t|re|ve|ll|d|m)\b", r"'\1", s)
    return _tok.findall(s)

def ngrams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]

FEAT_NAMES = ["nov1", "nov2", "nov3", "nov4", "num_nov", "len_tok", "n_sent", "rep2_in_sum", "rep3_in_sum",
              "ends_clean", "first_pos", "mean_pos", "lead3_cov", "lead_cov2", "art_cov1", "cap_ratio", "lcs_sent_max"]

def features(summary: str, article: str) -> np.ndarray:
    st, at = toks(summary), toks(article)
    out = []
    for n in (1, 2, 3, 4):
        sg = ngrams(st, n); ag = set(ngrams(at, n))
        out.append(1.0 - (sum(g in ag for g in sg) / len(sg)) if sg else 0.0)
    nums = [w for w in st if any(c.isdigit() for c in w)]
    aset = set(at)
    out.append(sum(w not in aset for w in nums) / max(1, len(nums)))
    out.append(np.log1p(len(st)))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s]
    out.append(len(sents))
    s2 = ngrams(st, 2); s3 = ngrams(st, 3)
    out.append(1.0 - len(set(s2)) / max(1, len(s2)))
    out.append(1.0 - len(set(s3)) / max(1, len(s3)))
    out.append(1.0 if summary.strip()[-1:] in ".!?'\"" else 0.0)
    # position of summary trigrams in article (lead bias)
    pos = {}
    for i, g in enumerate(ngrams(at, 3)):
        pos.setdefault(g, i)
    hits = [pos[g] / max(1, len(at)) for g in s3 if g in pos]
    out.append(hits[0] if hits else 1.0)
    out.append(float(np.mean(hits)) if hits else 1.0)
    asents = [s for s in re.split(r"(?<=[.!?])\s+", article.strip()) if s]
    lead = set(toks(" ".join(asents[:3])))
    out.append(len(lead & set(st)) / max(1, len(lead)))
    lead2 = set(ngrams(toks(" ".join(asents[:5])), 2))
    out.append(len(lead2 & set(s2)) / max(1, len(lead2)))
    out.append(len(set(st) & aset) / max(1, len(aset)))
    words = summary.split()
    out.append(sum(w[:1].isupper() for w in words) / max(1, len(words)))
    # max fraction of a summary sentence copied verbatim (as 4-grams)
    ag4 = set(ngrams(at, 4)); best = []
    for s in sents:
        g = ngrams(toks(s), 4)
        if g: best.append(sum(x in ag4 for x in g) / len(g))
    out.append(min(best) if best else 0.0)
    return np.array(out, dtype=np.float32)
