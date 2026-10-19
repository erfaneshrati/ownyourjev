"""Cross-process GPU slot scheduler (lock files), so several experiments can share the GPUs.

A job asks for ``need_gb`` of free memory. It takes the first slot lock on a GPU whose *current* free memory
is at least ``need_gb`` (other processes, e.g. an inference server, are respected) and holds the lock until
the context exits. Slots per GPU: ``OYJ_GPU_SLOTS`` ("0:3,1:2"), default 2 per GPU.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

LOCK_DIR = Path(os.environ.get("OYJ_GPU_LOCK_DIR", Path(__file__).resolve().parents[2] / "cache" / "gpu_locks"))


def _slots() -> dict[int, int]:
    import torch

    spec = os.environ.get("OYJ_GPU_SLOTS")
    n = torch.cuda.device_count()
    out = {i: 2 for i in range(n)}
    if spec:
        for part in spec.split(","):
            g, k = part.split(":")
            if int(g) < n:
                out[int(g)] = int(k)
    return out


def _free_gb(i: int) -> float:
    """Memory this process could use on GPU i: what is free plus what this process already reserves
    (a process holding cached memory from an earlier step must not wait for itself: that deadlocks)."""
    import torch

    try:
        free, _ = torch.cuda.mem_get_info(i)
        mine = torch.cuda.memory_reserved(i)
        return (free + mine) / 1e9
    except Exception:
        return 0.0


@contextlib.contextmanager
def gpu_slot(need_gb: float = 4.0, poll_s: float = 15.0, log=print):
    """Yield a device string ("cuda:i") with a held slot; waits until one is available."""
    import torch

    if not torch.cuda.is_available():
        yield "cpu"
        return
    import gc

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    waited = 0.0
    max_wait_s = float(os.environ.get("OYJ_GPU_MAX_WAIT_S", "120"))
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        # after max_wait_s, take any free slot regardless of memory (callers retry on out-of-memory)
        relax = waited >= max_wait_s
        for gpu, k in sorted(_slots().items(), key=lambda kv: -_free_gb(kv[0])):
            if not relax and _free_gb(gpu) < need_gb:
                continue
            for s in range(k):
                f = open(LOCK_DIR / f"gpu{gpu}_slot{s}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                f.write(f"{os.getpid()}\n")
                f.flush()
                try:
                    yield f"cuda:{gpu}"
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return
        if waited == 0 or int(waited) % 300 < poll_s:
            log(f"    waiting for a GPU slot with {need_gb:.0f} GB free ...")
        time.sleep(poll_s)
        waited += poll_s
