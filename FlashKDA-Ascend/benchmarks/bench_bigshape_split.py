"""Re-profile at the large shape, where the balance is probably different.

Everything so far was profiled at T=1024 H=8. That shape has kernel2 running at
blockDim = N*H = 8 against 20 cube cores, so it is latency-bound by
construction and launch overhead dominated (79%). The shape that matters is
T=8192 H=64, where blockDim is 64 and there are 512 chunks -- 2051 launches,
but also 64 blocks of real work behind each one.

At ~5.7 us per launch, 2051 launches is about 11.7 ms of a 94.5 ms run, so
launch overhead should be a small minority there and compute should dominate.
If so, the optimizations that pay at T=1024 H=8 are not the ones that pay here,
and the next move should be chosen from this table rather than the old one.

Splits kernel1 from kernel2 with FLASH_KDA_SKIP_K2, and estimates the launch
floor from the known per-launch cost and the launch count.
"""
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401

# Chunk size comes from the extension, not a local constant: a test that
# hardcodes it silently stops matching the kernel when the kernel changes.
from flash_kda import _C as _kda_C
CHUNK = _kda_C.CHUNK

DEV = torch.device("npu:0")
D = 128
US_PER_LAUNCH = 5.7e-3      # ms, measured from the empty-launch sweep

SKIP_K2 = os.environ.get("FLASH_KDA_SKIP_K2") == "1"


def bench(T, H, iters=3):
    from flash_kda import fwd, _C, clear_workspace_cache

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    a_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
    scale = 1.0 / math.sqrt(D)

    def call():
        fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0)

    for _ in range(2):
        call()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch_npu.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    del q, k, v, g, beta, out
    clear_workspace_cache()
    torch_npu.npu.empty_cache()
    return ms


def main():
    tag = "kernel1 only" if SKIP_K2 else "full pipeline"
    print(f"=== {tag} ===")
    print(f"{'T':>6} {'H':>4} {'chunks':>7} {'k1 blocks':>10} {'k2 blocks':>10} "
          f"{'launches':>9} {'launch ms':>10} {'total ms':>9} {'launch %':>9}")

    for T, H in [(2048, 64), (4096, 64), (8192, 64)]:
        chunks = T // CHUNK
        k1_blocks = chunks * H
        k2_blocks = H
        launches = 4 if SKIP_K2 else 4 + 4 * chunks + 3
        floor = launches * US_PER_LAUNCH
        ms = bench(T, H)
        print(f"{T:>6} {H:>4} {chunks:>7} {k1_blocks:>10} {k2_blocks:>10} "
              f"{launches:>9} {floor:>10.2f} {ms:>9.2f} {100 * floor / ms:>8.0f}%")

    print()
    print("kernel1 dispatches one block per (tile, head) -- at T=8192 H=64 that")
    print("is 32768 blocks per launch across 20 cores.")


if __name__ == "__main__":
    main()
