"""Which kernel is non-deterministic, and which unit inside it?

race_probe showed the final output changing run to run (7.9e-4 at T=64 H=4,
1.9e-3 at T=1024 H=8) while T=32 H=2 stayed bit-identical. It could not say
whether kernel1 or kernel2 owns the race, because comparing the whole workspace
is meaningless: most of it is scratch that no phase is required to define, so
the diff is dominated by regions that were never written.

This compares only the fields kernel1 is contractually required to produce,
for every (tile, head), with kernel2 skipped entirely:

    k_decayed, q_decayed, k_inv, k_restored, g_total   AIV outputs
    INV, Mqk                                            AIC outputs

If the AIV fields are stable but INV moves, the race is in the cube path
(the Neumann chain or Gemm16's barriers). If the AIV fields move, it is in the
vector phases. If everything here is stable, kernel1 is clean and kernel2's
state recurrence owns it.

Offsets mirror WorkspaceOffsets in layout.hpp, same as check_prepare.py.
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
CHUNK = 16
PER_TILE = 608256          # WorkspaceSizes::kPerTile

FIELDS = [
    ("k_decayed",  0,     CHUNK * D,     2),
    ("q_decayed",  4096,  CHUNK * D,     2),
    ("k_inv",      8192,  CHUNK * D,     2),
    ("k_restored", 12288, CHUNK * D,     2),
    ("g_total",    16384, D,             4),
    ("INV",        16896, CHUNK * CHUNK, 2),
    ("Mqk",        17408, CHUNK * CHUNK, 2),
    # The Neumann chain's inputs, which the first version of this probe missed:
    # slot 0 holds (I - L), slot 5 holds plain L, and kIdentity holds I. All
    # three are written by AIV and read by the AIC chain, so if any of them
    # moves, the corruption is upstream of the cube entirely.
    ("identity",   17920, CHUNK * CHUNK, 2),
    ("s0_ImL",     18432, CHUNK * CHUNK, 2),
    ("s5_L",       18432 + 5 * 65536, CHUNK * CHUNK, 2),
    # Slot 0 holds the FIRST Gemm128's fp32 output (L, 1024 bytes), which AIV
    # later overwrites in place with bf16 (I - L) -- but only the first 512
    # bytes. Bytes 512..1024 are still the raw cube result for rows 8..15, so
    # reading them says whether the cube itself produced garbage or whether the
    # AIV rewrite did. Slot 1 is the SECOND Gemm128's fp32 output and is never
    # overwritten, so it is the control.
    ("s0_cube_fp32", 18432 + 512,           128,           4),
    ("s1_cube_fp32", 18432 + 65536,   CHUNK * CHUNK,       4),
]
AIV = {"k_decayed", "q_decayed", "k_inv", "k_restored", "g_total"}


def extract(ws_u8, units):
    """[units, PER_TILE] uint8 -> {name: tensor}, all on host."""
    blocks = ws_u8[: units * PER_TILE].view(units, PER_TILE)
    out = {}
    for name, off, count, size in FIELDS:
        raw = blocks[:, off: off + count * size].contiguous()
        if size == 2:
            # bf16 is the top 16 bits of the fp32 pattern; torch can view it
            # directly, unlike struct's 'e' which is IEEE fp16.
            out[name] = raw.view(torch.bfloat16).float()
        else:
            out[name] = raw.view(torch.float32)
    return out


def main():
    from flash_kda import _C

    iters = 6
    print(f"{'T':>6} {'H':>3} {'units':>6} " +
          " ".join(f"{n:>11}" for n, _, _, _ in FIELDS))

    for T, H in [(32, 2), (64, 4), (256, 8), (1024, 8)]:
        torch.manual_seed(0)
        mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
        q, k, v, g = (mk(1, T, H, D) for _ in range(4))
        beta = mk(1, T, H)
        A_log = mk(H, dt=torch.float32)
        dt_bias = mk(H, D, dt=torch.float32)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
        scale = 1.0 / math.sqrt(D)
        units = (T // CHUNK) * H

        ref = None
        worst = {n: 0.0 for n, _, _, _ in FIELDS}
        for _ in range(iters):
            ws = torch.zeros(_C.get_workspace_size(T, H, 1),
                             dtype=torch.uint8).to(DEV)
            _C.fwd(q, k, v, g, beta, scale, out, ws, A_log, dt_bias, -5.0)
            torch_npu.npu.synchronize()
            cur = extract(ws.cpu(), units)
            if ref is None:
                ref = cur
            else:
                for n in worst:
                    d = (cur[n] - ref[n]).abs().max().item()
                    worst[n] = max(worst[n], d)
            del ws
            torch_npu.npu.empty_cache()

        print(f"{T:>6} {H:>3} {units:>6} " +
              " ".join(f"{worst[n]:>11.3e}" for n, _, _, _ in FIELDS))

    print()
    print("AIV fields:", " ".join(sorted(AIV)))
    print("AIC fields: INV Mqk")
    print("all zero -> kernel1 deterministic, race is kernel2's recurrence")


if __name__ == "__main__":
    main()
