"""Time the kernels with the workspace hoisted out of the loop.

Stubbing every phase of kernel1 left the time unchanged (27.96 ms vs 28.66 ms
full), which means almost none of it was compute. flash_kda.fwd allocates the
workspace on every call:

    workspace = torch.zeros(get_workspace_size(...), dtype=uint8).to(device)

At T=1024 H=8 that is 302 MB of host-side zeros copied to the device per call,
which at typical H2D bandwidth is tens of milliseconds -- the whole measurement.

This calls _C.fwd directly with a workspace allocated once, which is what any
real caller would do, and reports both numbers so the difference is visible.
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

DEV = torch.device("npu:0")
D = 128


def time_it(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    from flash_kda import fwd, _C

    print(f"{'T':>6} {'H':>4} {'workspace':>10} {'per-call alloc':>15} "
          f"{'hoisted':>9} {'alloc share':>12} {'us/tok/head':>12}")

    for T, H in [(512, 8), (1024, 8), (2048, 8), (2048, 16), (4096, 8)]:
        torch.manual_seed(0)
        mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
        q, k, v, g = (mk(1, T, H, D) for _ in range(4))
        beta = mk(1, T, H)
        a_log = mk(H, dt=torch.float32)
        dt_bias = mk(H, D, dt=torch.float32)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
        scale = 1.0 / math.sqrt(D)

        ws_bytes = _C.get_workspace_size(T, H, 1)
        try:
            ws = torch.zeros(ws_bytes, dtype=torch.uint8).to(DEV)
        except Exception as exc:  # noqa: BLE001
            print(f"{T:>6} {H:>4} {ws_bytes/2**20:>9.0f}M  cannot allocate: "
                  f"{type(exc).__name__}")
            continue

        with_alloc = time_it(lambda: fwd(q, k, v, g, beta, scale, out,
                                         a_log, dt_bias, -5.0))
        hoisted = time_it(lambda: _C.fwd(q, k, v, g, beta, scale, out, ws,
                                         a_log, dt_bias, -5.0))
        share = 100.0 * (with_alloc - hoisted) / with_alloc
        per = hoisted * 1e3 / (T * H)
        print(f"{T:>6} {H:>4} {ws_bytes/2**20:>9.0f}M {with_alloc:>15.2f} "
              f"{hoisted:>9.2f} {share:>11.0f}% {per:>12.4f}")

        del ws
        torch_npu.npu.empty_cache()

    print()
    print("CUDA reference on an H20, for scale (different hardware):")
    print("  T=8192 H=64 fixed  1.6217 ms  ->  0.0031 us/token/head")


if __name__ == "__main__":
    main()
