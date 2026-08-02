"""SUPERSEDED -- kept for the record.

Written to find the size at which the kernels started hanging. That hang was a
missing M_MTE1 barrier after Mmad ("L0B read/write conflict in the MTE"), fixed
long ago; both kernels now run every shape in the sweep. Superseded by
tests/test_shapes.py for correctness and tests/race_probe.py for the
intermittent faults this could not see.
"""

import math
import sys

import torch
import torch_npu  # registers the npu device

from flash_kda import fwd

DEV = torch.device("npu:0")


def case(B, T, H, D=128):
    torch.manual_seed(0)

    def mk(*shape, dt=torch.bfloat16):
        return torch.randn(*shape, dtype=dt).to(DEV)

    q, k, v, g = (mk(B, T, H, D) for _ in range(4))
    beta = mk(B, T, H)
    a_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=DEV)

    fwd(q, k, v, g, beta, 1.0 / math.sqrt(D), out, a_log, dt_bias, -5.0)

    o = out.cpu().float()
    return bool(torch.isfinite(o).all().item()), float(o.abs().sum())


CASES = [
    (1, 16, 1),   # 1 tile,  1 head  -> known good
    (1, 32, 1),   # 2 tiles, 1 head  -> does the chunk loop hang?
    (1, 16, 2),   # 1 tile,  2 heads -> does a second core hang?
    (1, 48, 1),
    (1, 64, 1),
    (1, 64, 4),
]

if __name__ == "__main__":
    for B, T, H in CASES:
        tiles = (T // 16) * H
        print(f"--> starting B={B} T={T} H={H} (cores={tiles})", flush=True)
        try:
            finite, total = case(B, T, H)
            print(f"    OK finite={finite} sum={total:.4e}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            sys.exit(1)
    print("all cases completed", flush=True)
