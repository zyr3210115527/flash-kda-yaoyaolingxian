"""Compare kernel2's chunk-0 intermediates against CPU.

Chunk 0 starts from zero state, so the recurrence collapses to

    v_sub = v * sigmoid(beta)
    u     = INV @ v_sub
    out   = Mqk @ u

with no state term at all. Mqk and INV are already verified correct in the
workspace by check_prepare, so any error here is in kernel2's use of them --
the beta application, or the INV @ v_sub and Mqk @ u GEMMs.

Slots, from the phase map:
    1  sigmoid(beta)        fp32 [16]
    2  v, then overwritten with v_sub   bf16 [16,128]
    0  u = INV @ v_sub      bf16 [16,128]
    8  Mqk @ u              fp32 [16,128]
    6  q_dec @ state        fp32 [16,128]   (zero for chunk 0)
"""
import math
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

CHUNK, D = 16, 128
OFF_INV, OFF_MQK = 16896, 17408
SCRATCH, SLOT = 18432, 65536


def slot(i):
    return SCRATCH + i * SLOT


def bf16(raw, off, n):
    b = struct.unpack_from(f'<{n}H', raw, off)
    f = struct.unpack(f'<{n}f', struct.pack(f'<{n}I', *(x << 16 for x in b)))
    return torch.tensor(f, dtype=torch.float64)


def f32(raw, off, n):
    return torch.tensor(struct.unpack_from(f'<{n}f', raw, off), dtype=torch.float64)


def rel(a, b):
    return ((a - b).flatten().square().mean().sqrt() /
            (b.flatten().square().mean().sqrt() + 1e-30)).item()


def main():
    import torch_npu  # noqa: F401
    from flash_kda import _C

    dev = torch.device("npu:0")
    B, T, H = 1, CHUNK, 1          # single chunk, single head: isolate
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ws = torch.zeros(_C.get_workspace_size(T, H, 1), dtype=torch.uint8).to(dev)
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=dev)
    _C.fwd(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), scale,
           out, ws, A_log.to(dev), dt_bias.to(dev), -5.0)
    torch_npu.npu.synchronize()
    raw = ws.cpu().numpy().tobytes()

    # Device-side values
    d_inv = bf16(raw, OFF_INV, CHUNK * CHUNK).reshape(CHUNK, CHUNK)
    d_mqk = bf16(raw, OFF_MQK, CHUNK * CHUNK).reshape(CHUNK, CHUNK)
    d_bsig = f32(raw, slot(1), CHUNK)
    d_vsub = bf16(raw, slot(2), CHUNK * D).reshape(CHUNK, D)
    d_u = bf16(raw, slot(0), CHUNK * D).reshape(CHUNK, D)
    d_mqku = f32(raw, slot(8), CHUNK * D).reshape(CHUNK, D)
    d_qs = f32(raw, slot(6), CHUNK * D).reshape(CHUNK, D)

    # CPU expectations, using the DEVICE's own Mqk and INV so this isolates
    # kernel2 from any kernel1 error.
    e_bsig = torch.sigmoid(beta[0, :, 0].double())
    e_vsub = v[0, :, 0].double() * e_bsig.unsqueeze(-1)
    e_u = d_inv @ e_vsub
    e_mqku = d_mqk @ e_u

    print("chunk 0, zero state -- kernel2 intermediates vs CPU:")
    print(f"  sigmoid(beta)  rel = {rel(d_bsig, e_bsig):.4e}")
    print(f"  v_sub          rel = {rel(d_vsub, e_vsub):.4e}")
    print(f"  u = INV@v_sub  rel = {rel(d_u, e_u):.4e}")
    print(f"  Mqk@u          rel = {rel(d_mqku, e_mqku):.4e}")
    print(f"  q_dec@state    max|.| = {d_qs.abs().max():.4e}   (want 0)")
    print()
    print(f"  final out      rel = {rel(out.cpu()[0, :, 0].double(), e_mqku):.4e}")


if __name__ == "__main__":
    main()
