"""SUPERSEDED -- kept for the record, do not trust its numbers.

This harness calls the `flash_kda.fwd` wrapper, which at the time allocated the
workspace on every call: 302 MB of host-side zeros copied to the device per
invocation at T=1024 H=8. That was 88-98% of everything it measured, and it is
where the "790x slower than CUDA" figure came from. The wrapper now caches the
workspace, so the allocation is gone, but this file is left as the artifact that
made the mistake visible.

Use benchmarks/ab_compare.py (interleaved A/B, medians, resolves ~1%) or
benchmarks/bench_cuda_shapes.py (the reference shapes) instead.
"""

import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401  -- registers the npu device

DEV = torch.device("npu:0")
D = 128


def workspace_bytes(T, H, N=1):
    from flash_kda._C import get_workspace_size
    return get_workspace_size(T, H, N)


def bench(T, H, B=1, warmup=3, iters=10):
    from flash_kda import fwd

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(B, T, H, D) for _ in range(4))
    beta = mk(B, T, H)
    a_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=DEV)
    scale = 1.0 / math.sqrt(D)

    for _ in range(warmup):
        fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0)
    torch_npu.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0)
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3     # ms


SHAPES = [
    (128, 4), (256, 4), (512, 4),
    (512, 8), (1024, 8),
    (2048, 8), (2048, 16),
]

if __name__ == "__main__":
    print(f"{'T':>6} {'H':>4} {'chunks':>7} {'k2 launches':>12} "
          f"{'workspace':>11} {'ms':>9} {'us/token/head':>14}")
    for T, H in SHAPES:
        ws = workspace_bytes(T, H, 1)
        chunks = T // 16
        launches = 2 + 5 * chunks
        if ws > 40 * 2**30:
            print(f"{T:>6} {H:>4} {chunks:>7} {launches:>12} "
                  f"{ws/2**30:>10.1f}G  SKIPPED (workspace)")
            continue
        try:
            ms = bench(T, H)
            per = ms * 1e3 / (T * H)
            print(f"{T:>6} {H:>4} {chunks:>7} {launches:>12} "
                  f"{ws/2**20:>10.0f}M {ms:>9.3f} {per:>14.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{T:>6} {H:>4} {chunks:>7} {launches:>12} "
                  f"{ws/2**20:>10.0f}M  ERROR {type(exc).__name__}: {str(exc)[:40]}")

    print()
    print("For scale, the CUDA version on an H20 (different hardware, so this is")
    print("an order-of-magnitude reference only):")
    print("  T=8192 H=64  D=128  fixed   1.6217 ms   -> 0.0031 us/token/head")
    print("  T=8192 H=96  D=128  fixed   2.6220 ms   -> 0.0033 us/token/head")
