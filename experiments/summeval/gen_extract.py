"""Programmatic extractive summaries (in the style of SummEval's extractive systems) for real or new articles.
usage: gen_extract.py <tag> real|new"""
import random, re, sys
from common import *
from gen_synth import real_articles, ptb_style
from ownyourjev.data import Example, write_jsonl, read_jsonl

SPL = re.compile(r"(?<=[.!?'\"])\s+(?=[A-Z0-9'\"`(])")


def variants(sents, rng):
    n = len(sents)
    mid = list(range(min(3, n), n))
    out = {"lead3": sents[:3], "lead2": sents[:2]}
    if n > 4:
        r = sorted(rng.sample(range(n), 3)); out["rand3_sorted"] = [sents[i] for i in r]
        r = rng.sample(range(n), 3); out["rand3_shuffled"] = [sents[i] for i in r]
        out["first_plus_rand"] = [sents[0]] + [sents[i] for i in sorted(rng.sample(range(1, n), 2))]
    if len(mid) >= 3:
        out["late3"] = [sents[i] for i in sorted(rng.sample(mid, 3))]
    out["lead3_dup"] = sents[:2] + [sents[0]] if n >= 2 else sents[:1]
    return out


def main(tag, which):
    if which == "real":
        arts = real_articles()
    else:
        arts = [(e.group, e.state["source_document"], -2) for e in read_jsonl(DATA / "new_articles.jsonl")]
    out = []
    for g, a, f in arts:
        rng = random.Random(f"ext-{g}")
        sents = [s.strip() for s in SPL.split(a) if len(s.strip()) > 20]
        for name, ss in variants(sents, rng).items():
            text = " ".join(ss)
            if name == "lead3" and rng.random() < 0.5:  # truncated like length-capped systems
                w = text.split(); text = " ".join(w[: rng.randint(35, 60)])
            text = ptb_style(text, rng) if rng.random() < 0.85 else text
            out.append(Example(f"ext-{tag}-{g}-{name}", {"source_document": a, "summary": text}, None, None, g,
                               {"synthetic": True, "profile": "ext_" + name, "fold": f}))
    write_jsonl(DATA / f"synth_{tag}.jsonl", out); print(len(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
