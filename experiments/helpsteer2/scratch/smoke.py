import asyncio, time
from common import *
from label_dec import label
from ownyourjev.llm import OpenRouterClient
async def m():
    c = OpenRouterClient(model="qwen/qwen3.7-flash", concurrency=8, timeout=60)
    hold = {e.uid for e in protocol.dev(T)}
    p = [e for e in protocol.pool(T) if e.uid not in hold][7000:7008]
    t = time.time()
    out = await label(c, [Example(e.uid, e.state, None, None, e.group, {}) for e in p], f"{T}:pool_dec_qwen")
    print([e.labels is not None for e in out], [e.meta.get("label_error") for e in out], time.time()-t, c.stats)
    await c.aclose()
asyncio.run(m())
