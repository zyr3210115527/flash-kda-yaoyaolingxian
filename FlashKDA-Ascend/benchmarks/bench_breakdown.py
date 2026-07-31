"""Break the Ascend forward time down: kernel1 vs kernel2 vs launch overhead.

The aggregate is ~790x the CUDA reference per token-head, which on its own does
not say what to fix. This separates:

  - kernel1 alone            (4 launches, FLASH_KDA_SKIP_K2=1)
  - full pass                (4 + 2 + 5*chunks launches)
  - an empty kernel launch   (_C.noop, to price the launch itself)

so the fraction that is launch overhead versus actual compute is visible.
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


def launch_cost():
    """Cost of one trivial kernel launch, from _C.noop."""
    from flash_kda import _C
    flag = torch.zeros(8, dtype=torch.int32).to(DEV)
    n = 200
    def burst():
        for _ in range(n):
            _C.noop(flag)
    ms = time_it(burst, warmup=2, iters=5)
    return ms / n * 1e3      # us per launch


def main():
    from flash_kda import fwd

    per_launch_us = launch_cost()
    print(f"empty kernel launch: {per_launch_us:.1f} us\n")

    print(f"{'T':>6} {'H':>4} {'k1 ms':>8} {'full ms':>9} {'k2 ms':>8} "
          f"{'launches':>9} {'launch ms':>10} {'launch %':>9}")

    for T, H in [(512, 8), (1024, 8), (2048, 8)]:
        torch.manual_seed(0)
        mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
        q, k, v, g = (mk(1, T, H, D) for _ in range(4))
        beta = mk(1, T, H)
        a_log = mk(H, dt=torch.float32)
        dt_bias = mk(H, D, dt=torch.float32)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
        scale = 1.0 / math.sqrt(D)
        call = lambda: fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0)

        os.environ['FLASH_KDA_SKIP_K2'] = '1'
        k1 = time_it(call)
        del os.environ['FLASH_KDA_SKIP_K2']
        full = time_it(call)

        chunks = T // 16
        launches = 4 + 2 + 5 * chunks
        lms = launches * per_launch_us / 1e3
        print(f"{T:>6} {H:>4} {k1:>8.2f} {full:>9.2f} {full-k1:>8.2f} "
              f"{launches:>9} {lms:>10.2f} {100*lms/full:>8.0f}%")


if __name__ == "__main__":
    main()
