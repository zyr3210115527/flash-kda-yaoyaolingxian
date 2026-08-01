"""Which elements of the first Gemm128's fp32 output are garbage?

Established so far:
  - every AIV field is bit-identical across runs
  - slot 1, the SECOND Gemm128's fp32 output, is bit-identical
  - slot 0, the FIRST Gemm128's fp32 output, varies by ~1e13 in rows 8..15
  - swapping the two calls moves the corruption to whichever runs first

1e13 is not a rounding difference, it is uninitialized memory being read out.
So the first cube op writes only part of its destination and the rest is
whatever was there before.

This dumps slot 0 as raw fp32 for every unit and reports, per element position,
whether it is stable across runs and whether it is plausible as a value of
L = k_decayed @ k_inv^T (entries are O(1) after L2 normalization, so anything
above ~1e3 is garbage regardless of stability).

Run with FLASH_KDA_SKIP_K2=1. Note MaskAndBuild later overwrites bytes 0..512
of this slot with bf16 (I - L), so the first 8 rows read here are that bf16
data reinterpreted as fp32 and are expected to look odd; rows 8..15 are the
untouched cube output and are the interesting part.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch_npu  # noqa: F401

# Chunk size comes from the extension, not a local constant: a test that
# hardcodes it silently stops matching the kernel when the kernel changes.
from flash_kda import _C as _kda_C
CHUNK = _kda_C.CHUNK

DEV = torch.device("npu:0")
D = 128
PER_TILE = 608256
SCRATCH = 18432
SLOT = 65536


def slot_fp32(ws_u8, units, which):
    off = SCRATCH + which * SLOT
    blocks = ws_u8[: units * PER_TILE].view(units, PER_TILE)
    raw = blocks[:, off: off + CHUNK * CHUNK * 4].contiguous()
    return raw.view(torch.float32).view(units, CHUNK, CHUNK)


def main():
    from flash_kda import _C

    T, H = 64, 4
    iters = 5
    units = (T // CHUNK) * H

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    A_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
    scale = 1.0 / math.sqrt(D)

    runs0, runs1 = [], []
    for _ in range(iters):
        ws = torch.zeros(_C.get_workspace_size(T, H, 1),
                         dtype=torch.uint8).to(DEV)
        _C.fwd(q, k, v, g, beta, scale, out, ws, A_log, dt_bias, -5.0)
        torch_npu.npu.synchronize()
        h = ws.cpu()
        runs0.append(slot_fp32(h, units, 0))
        runs1.append(slot_fp32(h, units, 1))
        del ws
        torch_npu.npu.empty_cache()

    for name, runs in (("slot 0  (Gemm128 #1, L)", runs0),
                       ("slot 1  (Gemm128 #2, Mqk)", runs1)):
        ref = runs[0]
        unstable = torch.zeros(CHUNK, CHUNK, dtype=torch.bool)
        for r in runs[1:]:
            unstable |= (r != ref).any(dim=0)
        huge = (ref.abs() > 1e3).any(dim=0)

        print(f"\n{name}")
        print("  unstable across runs ('#'), by (row, col):")
        for i in range(CHUNK):
            print("    " + "".join("#" if unstable[i, j] else "." for j in range(CHUNK)))
        print(f"  unstable positions: {int(unstable.sum())}/{CHUNK * CHUNK}")
        print(f"  |value| > 1e3:      {int(huge.sum())}/{CHUNK * CHUNK}")
        print(f"  rows 8..15 unstable: {int(unstable[8:].sum())}/{8 * CHUNK}")
        print(f"  rows 0..7  unstable: {int(unstable[:8].sum())}/{8 * CHUNK}")
        finite = ref[ref.abs() < 1e3]
        print(f"  plausible entries: {finite.numel()}/{ref.numel()}, "
              f"max |v| = {finite.abs().max().item() if finite.numel() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
