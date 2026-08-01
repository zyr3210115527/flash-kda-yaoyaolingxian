"""Is kernel2's launch overhead host-side or device-side?

Stubbing every kernel2 phase and varying only how many launches the host issues
per chunk gives a perfectly linear result at T=4096 H=64:

    4 launches/chunk   36.56, 37.82 ms
    2 launches/chunk   25.96, 26.90
    1 launch /chunk    21.03, 21.31
    kernel1 alone      15.86

Subtracting kernel1, that is ~5.2 us per launch at every count -- 21 ms of a
40 ms pipeline spent on 4096 launches that do nothing.

Which end pays matters, because the fixes are completely different:

  host-bound    the CPU cannot issue <<<>>> fast enough and the device starves.
                Fixed by issuing fewer API calls -- graph capture, or moving the
                chunk loop into the kernel. Merging phases helps only because it
                means fewer calls.

  device-bound  each launch has real cost on the device (block setup, pipeline
                drain at the kernel boundary). Fixed only by having fewer
                kernels, which means solving the AIC/AIV handoff.

The test: time the call with and without a synchronize. Everything is issued
asynchronously to the stream, so if the un-synchronized call already takes the
full time, the host is the bottleneck. If it returns immediately and the time
appears at the synchronize, the device is.
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

# Chunk size comes from the extension, not a local constant: a test that
# hardcodes it silently stops matching the kernel when the kernel changes.
from flash_kda import _C as _kda_C
CHUNK = _kda_C.CHUNK

DEV = torch.device("npu:0")
D = 128
T, H = 4096, 64


def main():
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

    issue, total = [], []
    for _ in range(7):
        torch_npu.npu.synchronize()
        t0 = time.perf_counter()
        call()
        t1 = time.perf_counter()          # host has finished issuing
        torch_npu.npu.synchronize()
        t2 = time.perf_counter()          # device has finished executing
        issue.append((t1 - t0) * 1e3)
        total.append((t2 - t0) * 1e3)

    mi, mt = statistics.median(issue), statistics.median(total)
    launches = 4 + 4 * (T // CHUNK) + 3
    print(f"T={T} H={H}, {launches} launches")
    print(f"  host issue time   {mi:7.2f} ms   ({mi * 1000 / launches:5.2f} us/launch)")
    print(f"  total wall time   {mt:7.2f} ms")
    print(f"  device-only tail  {mt - mi:7.2f} ms")
    print()
    if mi > 0.6 * mt:
        print("HOST-BOUND: the CPU cannot issue launches fast enough and the")
        print("device starves. Fewer API calls is the fix -- graph capture, or")
        print("the chunk loop inside the kernel.")
    elif mi < 0.2 * mt:
        print("DEVICE-BOUND: launches are cheap to issue but cost real time on")
        print("the device. Only fewer kernels helps, which means solving the")
        print("AIC/AIV handoff.")
    else:
        print("MIXED: neither end dominates; both matter.")

    clear_workspace_cache()


if __name__ == "__main__":
    main()
