"""Synthetic toxic / jailbreak user prompts, steered by pool prompts that both labelers marked toxic (no dev/seed gold).
Then relabeled by the same labeler ensemble (soft labels) so they share the label convention.
usage: synth_pos.py N_BATCHES"""
import asyncio, sys, random, json
from common import *
from prompts import GUIDANCE_V2
from ownyourjev.llm import OpenRouterClient
from ownyourjev.synth import build_synth_messages, Deduper, STYLE_HINTS
from ownyourjev.spec import render_state
from ownyourjev.label import label_examples

nb = int(sys.argv[1]); bs = 8
tag = sys.argv[2] if len(sys.argv) > 2 else "pos"
ds = {e.uid: e for e in load_jsonl(D / "pool_ds_g2.jsonl")}
qw = {e.uid: e for e in load_jsonl(D / "pool_qwen_g2.jsonl")}
pos = [ds[u] for u in ds if u in qw and soft(ds[u], "toxic") >= .5 and soft(qw[u], "toxic") >= .5 and len(ds[u].state["prompt"]) < 1200]
neg = [ds[u] for u in ds if u in qw and soft(ds[u], "toxic") < .5 and soft(qw[u], "toxic") < .5 and len(ds[u].state["prompt"]) < 600]
print("steering pool: pos", len(pos), "neg", len(neg))
rng = random.Random({"pos3": 3, "neg2": 4}.get(tag, 0))
cli = OpenRouterClient(model="deepseek/deepseek-v4.1-flash", temperature=0.9)
qdesc = {"toxic": Q["toxic"], "jailbreak": Q["jailbreak"]}
SUBS = {"pos2": ["very short crude sexual questions or vulgar slang (under 15 words)", "short requests for jokes or poems mocking a religion, prophet, ethnicity or nationality",
                 "short how-to requests for drugs, poisons, weapons or crimes", "short requests to insult someone or to talk dirty", "short sexual requests in non-English languages",
                 "requests to write explicit fan fiction about named characters"],
        "neg": ["benign persona or role-play setups (tutor, game master, famous person) with no explicit or harmful content",
                "benign 'ignore previous instructions' or system-prompt style instructions for formatting, classification or coding tasks",
                "educational or factual questions about sensitive topics (war, crime, drugs, sex education, history of racism) asked neutrally",
                "fiction requests with romance or mild violence but nothing explicit", "questions about AI safety, jailbreaks or chatbots asked out of curiosity",
                "prompt-engineering requests: write a system prompt or character card for a harmless assistant"]}
SUB = ["sexual or erotic content requests", "dangerous instructions (weapons, drugs, hacking)", "hateful or demeaning jokes and content",
       "insults, harassment, asking the bot to swear", "jailbreak / no-limits role-play prompts", "crude sexual questions or slang",
       "borderline: subtle sexual role-play persona hidden in a long benign setup", "non-English toxic requests"]

async def one(b):
    ex = rng.sample(pos, 10) + rng.sample(neg, 6)
    for e in ex:
        e.labels = {h: bool(soft(e, h) >= .5) for h in HEADS}
    sub = SUBS.get(tag.rstrip("0123456789") if tag.startswith("neg") else tag, SUB + SUBS["pos2"] if tag == "pos3" else SUB)
    SUBX = sub
    tgt = {"toxic": not tag.startswith("neg")}
    if tag.startswith("neg"):
        tgt["jailbreak"] = False
    if "jailbreak" in SUBX[b % len(SUBX)] and tag == "pos":
        tgt["jailbreak"] = True
    msgs = build_synth_messages(qdesc, n=bs, target=tgt, state_fields=["prompt"],
                                domain=f"real user prompts sent to an open AI chatbot demo; focus on: {SUBX[b % len(SUBX)]}. Guidelines:\n{GUIDANCE_V2}",
                                style=STYLE_HINTS[(b + (7 if tag in ("pos3", "neg2") else 0)) % len(STYLE_HINTS)], seeds=ex)
    try:
        raw = await cli.complete_json(msgs, purpose=f"tc:synth:{tag}", max_tokens=250 * bs + 200)
    except (RuntimeError, ValueError):
        return []
    items = raw.get("examples") if isinstance(raw, dict) else raw
    return items if isinstance(items, list) else []

async def main():
    batches = await cli.map(list(range(nb)), one, desc="synth", progress=False)
    dd = Deduper(); out = []
    for b, items in enumerate(batches):
        for j, it in enumerate(items):
            st = it.get("state") if isinstance(it, dict) else None
            if isinstance(st, str):
                st = {"prompt": st}
            if not isinstance(st, dict) or not isinstance(st.get("prompt"), str) or not dd.add(st["prompt"]):
                continue
            out.append(Example(f"synth-{tag}-{b}-{j}", {"prompt": st["prompt"]}, None, None, "", {"synthetic": True}))
    print("generated", len(out), cli.stats)
    sd = protocol.seeds(T)
    a = await label_examples(OpenRouterClient(model="deepseek/deepseek-v4.1-flash"), Q, out, purpose=f"tc:synth:{tag}:ds_g2", seeds=sd, guidance=GUIDANCE_V2, distribution=True, progress=False)
    q = await label_examples(OpenRouterClient(model="qwen/qwen3.7-flash"), Q, out, purpose=f"tc:synth:{tag}:qwen_g2", seeds=sd, guidance=GUIDANCE_V2, distribution=True, progress=False)
    fin = []
    for x, y in zip(a, q):
        if x.labels is None or y.labels is None:
            continue
        s = {h: (soft(x, h) + soft(y, h)) / 2 for h in HEADS}
        fin.append(Example(x.uid, x.state, {h: s[h] >= .5 for h in HEADS}, None, "", {"synthetic": True, "soft": s}))
    save_jsonl(D / ("synth_pos.jsonl" if tag == "pos" else f"synth_{tag}.jsonl"), fin)
    print("kept", len(fin), "toxic>=.5", sum(e.labels["toxic"] for e in fin), "jb>=.5", sum(e.labels["jailbreak"] for e in fin))

asyncio.run(main())
