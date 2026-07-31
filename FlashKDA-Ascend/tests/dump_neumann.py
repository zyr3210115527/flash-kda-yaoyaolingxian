"""Dump kernel1's Neumann intermediates and compare each against CPU.

Mqk passes, so the AIC GEMM path is sound. INV is wrong. The chain is

    slot5 = L                     (written by the AIV)
    slot0 = I - L                 (written by the AIV)
    ident = identity              (written by the AIV)
    slot2 = L^2                   then L^4, then L^8
    slot4 = P, the running product
    kINV  = final

Comparing each slot against a float64 CPU computation says exactly where it
diverges, instead of guessing at the accumulate semantics.
"""
import math
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHUNK = 16
D = 128

# WorkspaceOffsets from layout.hpp
OFF = {
    'k_decayed': 0, 'q_decayed': 4096, 'k_inv': 8192, 'k_restored': 12288,
    'g_total': 16384, 'INV': 16896, 'Mqk': 17408, 'identity': 17920,
}
SCRATCH = 18432
SLOT = 65536


def slot(i):
    return SCRATCH + i * SLOT


def bf16_at(raw, off, count):
    bits = struct.unpack_from(f'<{count}H', raw, off)
    f = struct.unpack(f'<{count}f', struct.pack(f'<{count}I', *(b << 16 for b in bits)))
    return torch.tensor(f, dtype=torch.float64)


def rel(a, b):
    return ((a - b).flatten().square().mean().sqrt() /
            (b.flatten().square().mean().sqrt() + 1e-30)).item()


def main():
    import torch_npu  # noqa: F401
    from flash_kda import _C
    from check_prepare import gate_and_decay

    dev = torch.device("npu:0")
    T, H = CHUNK, 1
    torch.manual_seed(0)
    q = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    ws = torch.zeros(_C.get_workspace_size(T, H, 1), dtype=torch.uint8).to(dev)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)
    _C.fwd(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), scale,
           out, ws, A_log.to(dev), dt_bias.to(dev), lower_bound)
    torch_npu.npu.synchronize()
    raw = ws.cpu().numpy().tobytes()

    # --- CPU reference chain ---
    d = gate_and_decay(q[0, :, 0], k[0, :, 0], g[0, :, 0], None,
                       A_log[0], dt_bias[0], scale, lower_bound)
    L_full = d['k_decayed'] @ d['k_inv'].t()
    bs = torch.sigmoid(beta[0, :, 0].double()).unsqueeze(-1)
    L = torch.tril(L_full, diagonal=-1) * bs
    I = torch.eye(CHUNK, dtype=torch.float64)

    L2 = L @ L
    L4 = L2 @ L2
    L8 = L4 @ L4
    P0 = I - L
    P1 = P0 @ (I + L2)
    P2 = P1 @ (I + L4)
    P3 = P2 @ (I + L8)

    want = {
        'identity (ws)': (bf16_at(raw, OFF['identity'], CHUNK * CHUNK).reshape(CHUNK, CHUNK), I),
        'slot0  I - L ': (bf16_at(raw, slot(0), CHUNK * CHUNK).reshape(CHUNK, CHUNK), P0),
        'slot5  L     ': (bf16_at(raw, slot(5), CHUNK * CHUNK).reshape(CHUNK, CHUNK), L),
        'slot2  L^8   ': (bf16_at(raw, slot(2), CHUNK * CHUNK).reshape(CHUNK, CHUNK), L8),
        'slot4  P final': (bf16_at(raw, slot(4), CHUNK * CHUNK).reshape(CHUNK, CHUNK), P3),
        'kINV         ': (bf16_at(raw, OFF['INV'], CHUNK * CHUNK).reshape(CHUNK, CHUNK), P3),
    }

    print("Neumann chain, device vs CPU:")
    for name, (got, exp) in want.items():
        r = rel(got, exp)
        print(f"  {name}  rel RMS = {r:.3e}  {'PASS' if r < 5e-2 else 'FAIL'}")

    print()
    print(f"  |L| max        = {L.abs().max():.4e}")
    print(f"  slot5 max      = {want['slot5  L     '][0].abs().max():.4e}")
    print(f"  identity trace = {want['identity (ws)'][0].diagonal().sum():.4f} (want 16)")
    print(f"  kINV diagonal[:4] = {want['kINV         '][0].diagonal()[:4].tolist()}")
    print(f"  want diagonal[:4] = {P3.diagonal()[:4].tolist()}")


if __name__ == "__main__":
    main()
