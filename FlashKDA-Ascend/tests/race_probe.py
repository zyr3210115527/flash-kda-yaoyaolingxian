"""Is the non-determinism in kernel1 or kernel2?

test_shapes reports a different error for the same shape on different runs
(T=64 H=4 gave 6.5e-3, 1.03e-2, 1.31e-2), which is a race rather than rounding.
Re-running the whole suite is a poor probe: it reseeds, reallocates, and runs
both kernels, so it cannot say where the race is.

This calls the kernel repeatedly on identical inputs inside one process and
diffs the outputs against the first call. Anything non-zero is a race, and the
workspace comparison says which kernel owns it:

  kernel1 writes INV and the other per-tile fields, then stops. If those differ
  between two calls on the same input, the race is in kernel1.

  If INV is stable but the final output is not, kernel1 is fine and the race is
  in kernel2's state recurrence.

The workspace is deliberately re-zeroed between calls so a difference means the
kernels disagreed, not that they read each other's leftovers. The second pass
skips the re-zeroing to test exactly that: whether the kernels depend on a clean
workspace, which would make the race a missing initialization rather than a
missing barrier.
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
ITERS = 8


def run(T, H, rezero):
    from flash_kda import _C

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    A_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ws_bytes = _C.get_workspace_size(T, H, 1)
    ws = torch.zeros(ws_bytes, dtype=torch.uint8).to(DEV)

    outs = []
    wss = []
    for _ in range(ITERS):
        if rezero:
            # A fresh allocation rather than ws.zero_(): aclnn has no uint8
            # ZerosLike on this build, and this is what the wrapper does anyway.
            ws = torch.zeros(ws_bytes, dtype=torch.uint8).to(DEV)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
        _C.fwd(q, k, v, g, beta, scale, out, ws, A_log, dt_bias, -5.0)
        torch_npu.npu.synchronize()
        # Cast on the host: this CANN build's aclnn Cast fails on device.
        outs.append(out.cpu().float())
        # bf16 view of the workspace: kernel1's per-tile fields live here
        # Unwritten and scratch regions can hold NaN, which would swamp a
        # plain max; compare NaN positions separately from values.
        wss.append(ws.view(torch.bfloat16).cpu().float())

    nz = torch.nan_to_num
    out_diff = max((o - outs[0]).abs().max().item() for o in outs[1:])
    ws_diff = max((nz(w) - nz(wss[0])).abs().max().item() for w in wss[1:])
    n_out = sum(1 for o in outs[1:] if not torch.equal(o, outs[0]))
    n_ws = sum(1 for w in wss[1:]
               if not torch.equal(nz(w), nz(wss[0]))
               or not torch.equal(w.isnan(), wss[0].isnan()))
    return out_diff, ws_diff, n_out, n_ws


def main():
    print(f"{'T':>5} {'H':>3} {'rezero':>7} {'out differs':>12} {'max out d':>11} "
          f"{'ws differs':>11} {'max ws d':>10}")
    for T, H in [(64, 4), (1024, 8), (32, 2)]:
        for rezero in (True, False):
            od, wd, no, nw = run(T, H, rezero)
            print(f"{T:>5} {H:>3} {str(rezero):>7} {no:>7}/{ITERS - 1} "
                  f"{od:>11.3e} {nw:>7}/{ITERS - 1} {wd:>10.3e}")
    print()
    print("ws differs  -> race is in kernel1 (it owns the workspace fields)")
    print("only out    -> kernel1 is deterministic, race is in kernel2")
    print("only when rezero=False -> kernels need a clean workspace "
          "(missing init, not a barrier)")


if __name__ == "__main__":
    main()
