"""Correctness check for the Ascend FlashKDA kernels.

This image ships the AscendC compiler but not the precompiled built-in operator
library, so every torch_npu compute op (Cast, Add, normal_) fails to launch.
Allocation and H2D/D2H copies still work, so all tensor generation and the
reference computation run on CPU, and the device is used only to hold inputs
for our own kernels and to hand results back.
"""

import os
import sys
import math

import torch
import torch_npu  # registers the npu device with torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("npu:0")


def err_ratio(actual, expected):
    a, e = actual.float(), expected.float()
    num = (a - e).flatten().square().mean().sqrt().item()
    den = e.flatten().square().mean().sqrt().item()
    return num / (den + 1e-8)


def make_inputs(B, T, H, D=128, seed=0):
    torch.manual_seed(seed)
    cpu = {
        "q": torch.randn(B, T, H, D, dtype=torch.bfloat16),
        "k": torch.randn(B, T, H, D, dtype=torch.bfloat16),
        "v": torch.randn(B, T, H, D, dtype=torch.bfloat16),
        "g": torch.randn(B, T, H, D, dtype=torch.bfloat16),
        "beta": torch.randn(B, T, H, dtype=torch.bfloat16),
        "A_log": torch.randn(H, dtype=torch.float32),
        "dt_bias": torch.randn(H, D, dtype=torch.float32),
    }
    npu = {name: t.to(DEV) for name, t in cpu.items()}
    return cpu, npu


def run_case(B, T, H, D=128, seed=0):
    sys.path.insert(0, os.path.join(HERE, "tests"))
    from torch_ref import torch_ref
    from flash_kda import fwd

    cpu, npu = make_inputs(B, T, H, D, seed)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    # Reference on CPU; torch_ref writes into `out` in place.
    ref_out = torch.empty(B, T, H, D, dtype=torch.bfloat16)
    torch_ref(cpu["q"], cpu["k"], cpu["v"], cpu["g"], cpu["beta"], scale,
              ref_out, cpu["A_log"], cpu["dt_bias"], lower_bound)

    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=DEV)
    fwd(npu["q"], npu["k"], npu["v"], npu["g"], npu["beta"], scale, out,
        npu["A_log"], npu["dt_bias"], lower_bound)

    actual = out.cpu()
    r = err_ratio(actual, ref_out)
    finite = bool(torch.isfinite(actual.float()).all().item())
    nonzero = bool((actual.float().abs().sum() > 0).item())
    print(f"B={B} T={T} H={H} D={D} | err_ratio={r:.4e} "
          f"finite={finite} nonzero={nonzero}")
    return r, finite, nonzero


if __name__ == "__main__":
    r, finite, nonzero = run_case(1, 64, 4)
    if not finite:
        print("FAIL: output has non-finite values")
        sys.exit(1)
    if not nonzero:
        print("FAIL: output is all zeros -- kernel likely never wrote anything")
        sys.exit(1)
    ok = r < 2e-2
    print("PASS" if ok else f"FAIL: err_ratio {r:.4e} exceeds 2e-2")
    sys.exit(0 if ok else 1)
