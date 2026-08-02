"""Where inside INV does the non-determinism land?

race_fields narrowed it to one field: every AIV output and the Mqk cube output
are bit-identical across runs, while INV moves by up to 6e-2. The magnitude
grows slowly with unit count (5.2e-2 at 16 units, 6.2e-2 at 512), which is what
a fixed per-unit failure probability looks like when you take a max over more
samples -- so this is a per-unit hazard, not a cross-unit one.

INV is produced by a chain of six cube round trips inside one core, so a
per-unit hazard means a missing barrier somewhere in that chain. Which barrier
depends on where the wrong values land:

  a whole 16x16 unit wrong        -> the chain read a stale operand
  one row or column wrong         -> a partial Fixpipe/DataCopy overlap
  strictly-lower triangle only    -> L or (I-L) was stale
  scattered single elements       -> operand still in flight during LoadData

This reports, over repeated runs, how many units differ at all, and for the
worst unit the exact mask of differing positions.
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
PER_TILE = _kda_C.WS_PER_TILE
INV_OFF = _kda_C.WS_OFF_KINV


def inv_all(ws_u8, units):
    blocks = ws_u8[: units * PER_TILE].view(units, PER_TILE)
    raw = blocks[:, INV_OFF: INV_OFF + CHUNK * CHUNK * 2].contiguous()
    return raw.view(torch.bfloat16).float().view(units, CHUNK, CHUNK)


def main():
    from flash_kda import _C

    T, H = 256, 8
    iters = 6
    units = (T // CHUNK) * H

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    A_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
    scale = 1.0 / math.sqrt(D)

    runs = []
    for _ in range(iters):
        ws = torch.zeros(_C.get_workspace_size(T, H, 1),
                         dtype=torch.uint8).to(DEV)
        _C.fwd(q, k, v, g, beta, scale, out, ws, A_log, dt_bias, -5.0)
        torch_npu.npu.synchronize()
        runs.append(inv_all(ws.cpu(), units))
        del ws
        torch_npu.npu.empty_cache()

    ref = runs[0]
    diff = torch.zeros(units, CHUNK, CHUNK)
    for r in runs[1:]:
        diff = torch.maximum(diff, (r - ref).abs())

    per_unit = diff.view(units, -1).max(dim=1).values
    bad = (per_unit > 0).nonzero().flatten()
    print(f"T={T} H={H}  {units} units, {iters} runs")
    print(f"units differing: {len(bad)}/{units}")
    if len(bad) == 0:
        print("stable this time -- rerun, the hazard is intermittent")
        return

    print(f"worst unit diff: {per_unit.max().item():.3e}")
    print(f"unit indices: {bad[:16].tolist()}{' ...' if len(bad) > 16 else ''}")
    print(f"  (unit = tile*H + head, H={H}; so heads "
          f"{sorted(set((b.item() % H) for b in bad))})")

    w = int(per_unit.argmax())
    m = diff[w]
    print(f"\ndifference mask for unit {w} (row, col), '.' = identical:")
    for i in range(CHUNK):
        print("  " + "".join("#" if m[i, j] > 0 else "." for j in range(CHUNK)))

    tri = torch.tril(torch.ones(CHUNK, CHUNK), diagonal=-1).bool()
    nz = m > 0
    print(f"\nnonzero positions: {int(nz.sum())}")
    print(f"  strictly lower triangle: {int((nz & tri).sum())}")
    print(f"  diagonal:                {int(nz.diag().sum())}")
    print(f"  upper triangle:          {int((nz & ~tri).sum()) - int(nz.diag().sum())}")

    print("\nreference INV row sums (should be finite, near 1 on diagonal):")
    print("  diag:", [f"{ref[w][i, i]:.3f}" for i in range(4)], "...")


if __name__ == "__main__":
    main()
