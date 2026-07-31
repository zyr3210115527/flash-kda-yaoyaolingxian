"""Validate torch_ref.py's algebra, independent of any NPU.

torch_ref is the oracle every future correctness comparison will be measured
against, so if its algebra is wrong the whole exercise is meaningless. Nothing
here needs hardware.

Three checks:

1. The Neumann series really inverts. torch_ref builds
   INV = (I-L)(I+L^2)(I+L^4)(I+L^8) for a strictly-lower-triangular L, which
   should equal (I+L)^-1 exactly for 16x16 (L is nilpotent, L^16 = 0).
   Verified in float64 so the identity is tested, not the fp16 rounding.

2. The chunked recurrence equals the naive definition. KDA in its unchunked
   form is a per-token delta rule:

       for each token t:
           k_t, q_t normalized; decay a_t = exp(g_t)
           S <- diag-scaled S, then rank-1 corrected by beta_t
           out_t = q_t @ S

   Implemented directly, in float64, and compared against torch_ref's chunked
   path. This is the check that would catch a wrong mask, a wrong decay
   placement, or a transposed state.

3. The tail-chunk path. A sequence length that is not a multiple of 16 must give
   the same answer as the equivalent full-length prefix.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'repo', 'FlashKDA-Ascend', 'tests'))

torch.set_printoptions(precision=6, sci_mode=False)


def check_neumann():
    """INV = (I-L)(I+L^2)(I+L^4)(I+L^8) must equal (I+L)^-1 for strictly lower L."""
    torch.manual_seed(0)
    n = 16
    L = torch.tril(torch.randn(n, n, dtype=torch.float64), diagonal=-1)

    I = torch.eye(n, dtype=torch.float64)
    INV = I - L
    L2 = L @ L
    INV = INV + INV @ L2
    L4 = L2 @ L2
    INV = INV + INV @ L4
    L8 = L4 @ L4
    INV = INV + INV @ L8

    residual = (INV @ (I + L) - I).abs().max().item()
    exact = torch.linalg.inv(I + L)
    vs_exact = (INV - exact).abs().max().item()
    print(f"[1] Neumann: max|INV(I+L) - I| = {residual:.3e}, "
          f"max|INV - inv(I+L)| = {vs_exact:.3e}")
    return residual < 1e-9


def naive_kda(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound):
    """Unchunked KDA in float64, written straight from the recurrence.

    Shapes: q,k,v,g [T,H,D]; beta [T,H]. Returns out [T,H,D].

    State is carried as [key, value] so out_t = q_t @ S.
    """
    T, H, D = q.shape
    q = q.double()
    k = k.double()
    v = v.double()
    g = g.double()
    beta = beta.double()

    # Same gate the kernels use: gv = lower_bound * sigmoid(exp(A_log) * (g + dt))
    a = torch.exp(A_log.double()).view(1, H, 1)
    gv = lower_bound * torch.sigmoid(a * (g + dt_bias.double().unsqueeze(0)))

    # L2 normalize q and k per token, per head
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    out = torch.zeros(T, H, D, dtype=torch.float64)
    for h in range(H):
        S = torch.zeros(D, D, dtype=torch.float64)  # [key, value]
        for t in range(T):
            decay = torch.exp(gv[t, h])             # [D], per key dim
            S = S * decay.unsqueeze(-1)             # decay each key row
            kt = k[t, h]
            vt = v[t, h]
            b = torch.sigmoid(beta[t, h])
            # delta rule: correct the value currently predicted for kt
            pred = kt @ S
            S = S + torch.outer(kt, (vt - pred) * b)
            out[t, h] = (q[t, h] * scale) @ S
    return out


def check_recurrence(T=32, H=2, D=128, seed=0):
    from torch_ref import torch_ref

    torch.manual_seed(seed)
    q = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    ref_out = torch.empty(1, T, H, D, dtype=torch.bfloat16)
    torch_ref(q, k, v, g, beta, scale, ref_out, A_log, dt_bias, lower_bound)

    naive = naive_kda(q[0], k[0], v[0], g[0], beta[0], scale,
                      A_log, dt_bias, lower_bound)

    a = ref_out[0].double()
    rel = (a - naive).flatten().square().mean().sqrt() / \
          (naive.flatten().square().mean().sqrt() + 1e-12)
    print(f"[2] chunked vs naive (T={T},H={H}): relative RMS = {rel.item():.4e}")
    return rel.item()


def check_tail(T=40, H=1, D=128, seed=1):
    """A length that is not a multiple of 16 exercises the padded tail chunk."""
    from torch_ref import torch_ref

    torch.manual_seed(seed)
    q = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)

    ref_out = torch.empty(1, T, H, D, dtype=torch.bfloat16)
    torch_ref(q, k, v, g, beta, scale, ref_out, A_log, dt_bias, -5.0)
    naive = naive_kda(q[0], k[0], v[0], g[0], beta[0], scale, A_log, dt_bias, -5.0)

    rel = (ref_out[0].double() - naive).flatten().square().mean().sqrt() / \
          (naive.flatten().square().mean().sqrt() + 1e-12)
    print(f"[3] tail chunk (T={T}, {T % 16} rows in last chunk): "
          f"relative RMS = {rel.item():.4e}")
    return rel.item()


if __name__ == "__main__":
    ok = check_neumann()
    print("    -> Neumann identity", "HOLDS" if ok else "FAILS")
    r2 = check_recurrence()
    r3 = check_tail()
    print()
    print("bf16 round-off alone puts the floor around 1e-2; a relative RMS well")
    print("above that means the chunked formulation and the naive one disagree")
    print("structurally, not numerically.")
