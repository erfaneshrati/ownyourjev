"""Label the whole pool with a labeler variant (soft labels). usage: label_pool.py NAME MODEL USE_SEEDS GUIDANCE [uids_file]"""
import asyncio, sys
from common import *
from prompts import *
from ownyourjev.label import label_examples
from ownyourjev.llm import OpenRouterClient
name, model, use_seeds, guidance = sys.argv[1:5]
pool = protocol.pool(T)
if len(sys.argv) > 5:
    keep = set(open(sys.argv[5]).read().split()); pool = [e for e in pool if e.uid in keep]
eb = None if model.startswith(("deepseek", "qwen")) else {"provider": {"sort": "throughput"}}
cli = OpenRouterClient(model=model, extra_body=eb, concurrency=32)
out = asyncio.run(label_examples(cli, Q, pool, purpose=f"tc:pool:{name}", seeds=protocol.seeds(T) if use_seeds == "1" else None,
                                 guidance=globals()[guidance], distribution=True, progress=False))
save_jsonl(D / f"pool_{name}.jsonl", out)
print(name, len(out), "fail", sum(e.labels is None for e in out), "pos", sum(bool(e.labels and e.labels['toxic']) for e in out), cli.stats)
