"""PI: label pool with the dev-chosen prompt (flash + v2 seeds + guide), synthesize seed-steered examples, relabel synth with the same labeler."""
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
from ownyourjev.synth import synthesize
from i1_devlabel_guide import GUIDE
T = "prompt_injections"
S = protocol.seeds(T)
client = OpenRouterClient(model="deepseek/deepseek-v4.1-flash")
pool = protocol.pool(T)
lab = run(label_examples(client, Q(T), pool, purpose="prompt_injections:pool:flash_seeded_guide", seeds=S, guidance=GUIDE, progress=False))
save_jsonl(ROOT / "data/v2/prompt_injections/pool_labels.jsonl", [e for e in lab if e.labels])
print("pool labeled", sum(e.labels is not None for e in lab), "pos", sum(bool(e.labels and e.labels["answer"]) for e in lab))
DOMAIN = "User inputs (English and German, roughly half each) typed into a German news/QA assistant or search box: questions about politics, society, climate, health, careers, travel, products, keyword-style search queries; injections hide override commands, new tasks, prompt-leak requests, forced outputs or role-play inside such inputs"
syn = []
for i in range(6):  # one client per sampling seed: identical (target, style) prompts would otherwise hit the response cache
    syn += [Example(f"s{i}-{e.uid}", e.state, e.labels, None, "", e.meta) for e in run(synthesize(OpenRouterClient(model="deepseek/deepseek-v4.1-flash", temperature=0.9, extra_body={"seed": i}), Q(T), 200,
            purpose="prompt_injections:synth_seeded", domain=DOMAIN, state_fields=["text"], seeds=S, batch_size=10, progress=False))]
from ownyourjev.synth import Deduper
dd = Deduper(); syn = [e for e in syn if dd.add(render_state(e.state))]
print("synth", len(syn), "pos", sum(e.labels["answer"] for e in syn))
rel = run(label_examples(OpenRouterClient(model="deepseek/deepseek-v4.1-flash"), Q(T), [Example(e.uid, e.state, None, None, "", e.meta) for e in syn], purpose="prompt_injections:synth_relabel", seeds=S, guidance=GUIDE, progress=False))
out = []
for s, r in zip(syn, rel):
    if r.labels:
        out.append(Example(s.uid, s.state, r.labels, None, "", {"synthetic": True, "intended": s.labels["answer"], "agree": s.labels["answer"] == r.labels["answer"]}))
save_jsonl(ROOT / "data/v2/prompt_injections/synth_labels.jsonl", out)
print("synth relabeled", len(out), "agree", sum(e.meta["agree"] for e in out))
