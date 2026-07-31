"""Does the kernel actually need a zeroed workspace?

`flash_kda.fwd` allocates a fresh zeroed workspace on every call:

    workspace = torch.zeros(get_workspace_size(...), dtype=uint8).to(device)

At T=1024 H=8 that is 302 MB of host-side zeros copied to the device per call,
and it is 88-98% of end-to-end time for any caller that does not hoist it. The
kernel itself is 2.81 ms; the wrapper turns that into 29.96 ms.

Caching the buffer only helps if it does not have to be re-zeroed, because the
host->device copy is the expensive part, not the allocation. So the question is
whether any slot is read before it is written.

The existing race_probe reused a workspace across calls and saw identical
results, but that proves nothing: the stale contents came from the same inputs,
so they were identical anyway. This poisons the workspace with a non-zero
pattern instead. If the output matches the zeroed run bit for bit, nothing is
read before being written and the buffer is safe to reuse.

0xFF as bf16 is NaN, which is the useful poison here: NaN propagates through
arithmetic, so anything that reads uninitialized workspace shows up as NaN in
the output rather than as a small perturbation that might hide.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
D = 128


def run_with(fill, T, H):
    from flash_kda import _C

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    A_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
    scale = 1.0 / math.sqrt(D)

    n = _C.get_workspace_size(T, H, 1)
    ws = torch.full((n,), fill, dtype=torch.uint8).to(DEV)
    _C.fwd(q, k, v, g, beta, scale, out, ws, A_log, dt_bias, -5.0)
    torch_npu.npu.synchronize()
    return out.cpu().float()


def main():
    print(f"{'T':>6} {'H':>4} {'poison':>8} {'max diff vs zeroed':>20} "
          f"{'NaNs':>8} {'verdict':>10}")
    ok = True
    for T, H in [(64, 4), (256, 8), (1024, 8)]:
        base = run_with(0x00, T, H)
        for fill in (0xFF, 0x5A):
            got = run_with(fill, T, H)
            nans = int(got.isnan().sum())
            diff = (got - base).abs().max().item()
            good = (diff == 0.0) and nans == 0
            ok &= good
            print(f"{T:>6} {H:>4} {hex(fill):>8} {diff:>20.3e} {nans:>8} "
                  f"{'same' if good else 'DIFFERS':>10}")

    print()
    if ok:
        print("No slot is read before it is written: the workspace can be "
              "cached and reused without re-zeroing.")
    else:
        print("Something reads the workspace before writing it. The zeroing is "
              "load-bearing and caching must re-zero (which is the expensive "
              "part, so it would not help).")


if __name__ == "__main__":
    main()
