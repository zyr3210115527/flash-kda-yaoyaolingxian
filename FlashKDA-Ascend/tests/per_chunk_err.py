"""Report err_ratio per chunk, not just overall.

The aggregate number hides where the error lives. If chunk 0 is accurate and
later chunks degrade, the fault is in the state carried between chunks; if every
chunk is equally wrong, it is in the per-chunk math.

Single-chunk runs give 0.8845 overall, and sqrt(3/4) = 0.866 is what you would
get from chunk 0 being right and the other three left at zero -- so chunk 0
already looks close. This measures that directly.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

CHUNK = 16


def rel(a, b):
    a, b = a.double(), b.double()
    d = (a - b).flatten().square().mean().sqrt()
    n = b.flatten().square().mean().sqrt()
    return (d / (n + 1e-12)).item()


def main():
    import torch_npu  # noqa: F401
    from torch_ref import torch_ref
    from flash_kda import fwd

    dev = torch.device("npu:0")
    B, T, H, D = 1, 64, 4, 128
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lb = -5.0

    ref = torch.empty(B, T, H, D, dtype=torch.bfloat16)
    torch_ref(q, k, v, g, beta, scale, ref, A_log, dt_bias, lb)

    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=dev)
    fwd(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), scale, out,
        A_log.to(dev), dt_bias.to(dev), lb)
    got = out.cpu()

    print(f"overall  rel = {rel(got, ref):.4e}")
    print("per chunk:")
    for c in range(T // CHUNK):
        s, e = c * CHUNK, (c + 1) * CHUNK
        gc, rc = got[0, s:e], ref[0, s:e]
        zero = gc.float().abs().sum().item() == 0.0
        print(f"  chunk {c} (tokens {s:>3}-{e-1:>3})  rel = {rel(gc, rc):.4e}"
              f"{'   [all zero]' if zero else ''}")

    print("per head, chunk 0:")
    for h in range(H):
        print(f"  head {h}  rel = {rel(got[0, :CHUNK, h], ref[0, :CHUNK, h]):.4e}")


if __name__ == "__main__":
    main()
