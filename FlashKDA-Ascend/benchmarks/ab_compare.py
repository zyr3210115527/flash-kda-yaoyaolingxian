"""A/B two builds at the large shape, with enough repeats to beat the noise.

Single measurements at T=8192 H=64 vary by about +/-8% run to run (the same
committed build measured 0.1481, 0.1529 and 0.1601 us/token/head in three
consecutive runs). That is wider than most of the differences worth chasing, so
comparing one run of build A against one run of build B says nothing -- which is
how I briefly concluded that removing barriers had made things slower.

This runs N alternating repeats within one process and reports median and
spread, so the comparison is between distributions rather than samples.
Alternating rather than batching guards against thermal drift over the run.

Usage: the .so under test is whatever is installed; pass a label as argv[1].
Results append to /tmp/ab_results.txt so two builds can be compared across
invocations.
"""
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
D = 128
T, H = 4096, 64          # half the reference shape: same per-token cost, half the wait
REPEATS = 9


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
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

    for _ in range(3):
        call()
    torch_npu.npu.synchronize()

    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(3):
            call()
        torch_npu.npu.synchronize()
        samples.append((time.perf_counter() - t0) / 3 * 1e3)

    samples.sort()
    med = statistics.median(samples)
    lo, hi = samples[0], samples[-1]
    print(f"{label:<28} median {med:7.2f} ms   min {lo:7.2f}   max {hi:7.2f}   "
          f"spread {100 * (hi - lo) / med:4.1f}%")

    with open("/tmp/ab_results.txt", "a") as fh:
        fh.write(f"{label}\t{med:.3f}\t{lo:.3f}\t{hi:.3f}\n")

    clear_workspace_cache()


if __name__ == "__main__":
    main()
