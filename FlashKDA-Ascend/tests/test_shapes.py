"""Exercise the shapes and features the Ascend port claims to support.

Everything so far has been B=1, T=64, H=4, no state, no varlen. This walks the
cases that have never run on device:

  - T not a multiple of 16 (tail chunk)
  - several head counts and sequence lengths
  - initial_state / final_state, bf16 and fp32
  - cu_seqlens with ragged segments

Reference is torch_ref on CPU, which tests/validate_ref.py verifies against an
independent float64 implementation. bf16 end to end puts the floor near 1e-2.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401  -- registers the npu device; must precede any use

TOL = 3e-2
DEV = torch.device("npu:0")
D = 128


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).flatten().square().mean().sqrt() /
            (b.flatten().square().mean().sqrt() + 1e-12)).item()


def inputs(B, T, H, seed=0):
    torch.manual_seed(seed)
    return dict(
        q=torch.randn(B, T, H, D, dtype=torch.bfloat16),
        k=torch.randn(B, T, H, D, dtype=torch.bfloat16),
        v=torch.randn(B, T, H, D, dtype=torch.bfloat16),
        g=torch.randn(B, T, H, D, dtype=torch.bfloat16),
        beta=torch.randn(B, T, H, dtype=torch.bfloat16),
        A_log=torch.randn(H, dtype=torch.float32),
        dt_bias=torch.randn(H, D, dtype=torch.float32),
    )


def run_case(B, T, H, *, state_dtype=None, seglens=None, seed=0):
    from torch_ref import torch_ref
    from flash_kda import fwd

    d = inputs(B, T, H, seed)
    scale, lb = 1.0 / math.sqrt(D), -5.0

    cu = None
    N = B
    if seglens is not None:
        bounds = [0]
        for L in seglens:
            bounds.append(bounds[-1] + L)
        cu = torch.tensor(bounds, dtype=torch.long)
        N = len(seglens)

    init = final = None
    init_d = final_d = None
    if state_dtype is not None:
        torch.manual_seed(seed + 99)
        init = (torch.randn(N, H, D, D, dtype=torch.float32) * 0.05).to(state_dtype)
        final = torch.empty(N, H, D, D, dtype=state_dtype)
        init_d = init.to(DEV)
        final_d = torch.empty(N, H, D, D, dtype=state_dtype, device=DEV)

    ref = torch.empty(B, T, H, D, dtype=torch.bfloat16)
    torch_ref(d['q'], d['k'], d['v'], d['g'], d['beta'], scale, ref,
              d['A_log'], d['dt_bias'], lb,
              initial_state=init, final_state=final, cu_seqlens=cu)

    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=DEV)
    fwd(d['q'].to(DEV), d['k'].to(DEV), d['v'].to(DEV), d['g'].to(DEV),
        d['beta'].to(DEV), scale, out, d['A_log'].to(DEV), d['dt_bias'].to(DEV), lb,
        initial_state=init_d, final_state=final_d,
        cu_seqlens=cu.to(DEV) if cu is not None else None)

    r_out = rel(out.cpu(), ref)
    r_state = rel(final_d.cpu(), final) if final_d is not None else None
    return r_out, r_state


CASES = [
    ("T=16  H=1   one chunk",        dict(B=1, T=16, H=1)),
    ("T=64  H=4   baseline",         dict(B=1, T=64, H=4)),
    ("T=48  H=2",                    dict(B=1, T=48, H=2)),
    ("T=128 H=8",                    dict(B=1, T=128, H=8)),
    ("T=80  H=1   tail 0",           dict(B=1, T=80, H=1)),
    ("T=40  H=2   tail 8 rows",      dict(B=1, T=40, H=2)),
    ("T=24  H=4   tail 8 rows",      dict(B=1, T=24, H=4)),
    ("B=2   T=32  H=2",              dict(B=2, T=32, H=2)),
    ("state bf16  T=32 H=2",         dict(B=1, T=32, H=2, state_dtype=torch.bfloat16)),
    ("state fp32  T=32 H=2",         dict(B=1, T=32, H=2, state_dtype=torch.float32)),
    ("varlen [16,24,8] H=1",         dict(B=1, T=48, H=1, seglens=[16, 24, 8])),
    ("varlen [32,32] H=2",           dict(B=1, T=64, H=2, seglens=[32, 32])),
]

if __name__ == "__main__":
    npass = nfail = 0
    for name, kw in CASES:
        try:
            r_out, r_state = run_case(**kw)
            ok = r_out < TOL and (r_state is None or r_state < TOL)
            extra = f"  state {r_state:.3e}" if r_state is not None else ""
            print(f"  {name:<28} out {r_out:.3e}{extra}   {'PASS' if ok else 'FAIL'}")
            npass += ok
            nfail += (not ok)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<28} ERROR {type(exc).__name__}: {str(exc)[:60]}")
            nfail += 1
    print()
    print(f"{npass}/{npass + nfail} shape/feature cases pass (tol {TOL})")
    sys.exit(0 if nfail == 0 else 1)
