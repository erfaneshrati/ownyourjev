"""Label N more PAWS pool texts (no v1 label) with the dev-chosen prompt: deepseek-v4.1-flash + v2 seeds (dev acc 0.875)."""
import sys, random
from common import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
T = "paws"; N = int(sys.argv[1])
have = set(load_v1(T, "silver_seeded"))
rest = [e for e in protocol.pool(T) if e.uid not in have]
random.Random(7).shuffle(rest)
lab = run(label_examples(OpenRouterClient(model="deepseek/deepseek-v4.1-flash"), Q(T), rest[:N], purpose="paws:pool_more:flash_seeded", seeds=protocol.seeds(T), progress=False))
ok = [e for e in lab if e.labels]
save_jsonl(ROOT / "data/v2/paws/v2_seeded_more.jsonl", ok)
print("labeled", len(ok), "pos", sum(e.labels["answer"] for e in ok))
