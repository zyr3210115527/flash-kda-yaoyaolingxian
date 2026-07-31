"""Validate torch_ref.py's algebra, independent of any NPU.

torch_ref is the oracle every future correctness comparison will be measured
against, so if its algebra is wrong the whole exercise is meaningless. Nothing
here needs hardware.

Checks:

1. The Neumann series really inverts. torch_ref builds
   INV = (I-L)(I+L^2)(I+L^4)(I+L^8) for a strictly-lower-triangular L, which
   should equal (I+L)^-1 exactly for 16x16 (L is nilpotent, L^16 = 0).
   Verified in float64 so the identity is tested, not the fp16 rounding.

2. The chunked recurrence equals the naive definition. KDA unchunked is a
   per-token delta rule; implemented directly in float64 and compared. This is
   the check that catches a wrong mask, wrong decay placement, or a transposed
   state.

3. Tail chunks. A length that is not a multiple of 16 must still be right.

4. initial_state / final_state. Confirms the layout convention at the GM
   boundary, which the Ascend kernel has to match.

5. Variable-length batches via cu_seqlens, including a ragged tail.

bf16 round-off puts the agreement floor around 1e-2. Anything well above that
is a structural disagreement, not numerical noise.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE), os.path.join(HERE, 'tests')):
    if os.path.exists(os.path.join(cand, 'torch_ref.py')):
        sys.path.insert(0, cand)
        break
else:
    sys.path.insert(0, os.path.join(HERE, 'repo', 'FlashKDA-Ascend', 'tests'))

TOL = 2e-2


def rel_rms(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).flatten().square().mean().sqrt() /
            (b.flatten().square().mean().sqrt() + 1e-12)).item()


def make(T, H, D=128, seed=0):
    torch.manual_seed(seed)
    return dict(
        q=torch.randn(1, T, H, D, dtype=torch.bfloat16),
        k=torch.randn(1, T, H, D, dtype=torch.bfloat16),
        v=torch.randn(1, T, H, D, dtype=torch.bfloat16),
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16),
        beta=torch.randn(1, T, H, dtype=torch.bfloat16),
        A_log=torch.randn(H, dtype=torch.float32),
        dt_bias=torch.randn(H, D, dtype=torch.float32),
        scale=1.0 / math.sqrt(D),
        lower_bound=-5.0,
    )


def naive_kda(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
              seg=None, state0=None):
    """Unchunked KDA in float64, straight from the recurrence.

    q,k,v,g [T,H,D]; beta [T,H]. State carried as [key, value] so
    out_t = (q_t * scale) @ S. `seg` is a list of (start, end) token ranges that
    each restart the state. Returns (out [T,H,D], final_state [len(seg),H,D,D]
    in [key, value] order).
    """
    T, H, D = q.shape
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))

    a = torch.exp(A_log.double()).view(1, H, 1)
    gv = lower_bound * torch.sigmoid(a * (g + dt_bias.double().unsqueeze(0)))

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    if seg is None:
        seg = [(0, T)]
    out = torch.zeros(T, H, D, dtype=torch.float64)
    finals = torch.zeros(len(seg), H, D, D, dtype=torch.float64)

    for si, (bos, eos) in enumerate(seg):
        for h in range(H):
            if state0 is not None:
                S = state0[si, h].double().clone()
            else:
                S = torch.zeros(D, D, dtype=torch.float64)
            for t in range(bos, eos):
                S = S * torch.exp(gv[t, h]).unsqueeze(-1)
                kt, vt = k[t, h], v[t, h]
                b = torch.sigmoid(beta[t, h])
                S = S + torch.outer(kt, (vt - kt @ S) * b)
                out[t, h] = (q[t, h] * scale) @ S
            finals[si, h] = S
    return out, finals


def check_neumann():
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
    vs_exact = (INV - torch.linalg.inv(I + L)).abs().max().item()
    ok = residual < 1e-9
    print(f"[1] Neumann identity      max|INV(I+L)-I| = {residual:.2e}  "
          f"vs torch.linalg.inv = {vs_exact:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_basic(T=32, H=2):
    from torch_ref import torch_ref
    d = make(T, H, seed=0)
    out = torch.empty(1, T, H, 128, dtype=torch.bfloat16)
    torch_ref(d['q'], d['k'], d['v'], d['g'], d['beta'], d['scale'], out,
              d['A_log'], d['dt_bias'], d['lower_bound'])
    naive, _ = naive_kda(d['q'][0], d['k'][0], d['v'][0], d['g'][0], d['beta'][0],
                         d['scale'], d['A_log'], d['dt_bias'], d['lower_bound'])
    r = rel_rms(out[0], naive)
    print(f"[2] chunked vs naive      T={T} H={H}   rel RMS = {r:.2e}  "
          f"{'PASS' if r < TOL else 'FAIL'}")
    return r < TOL


def check_tail(T=40, H=1):
    from torch_ref import torch_ref
    d = make(T, H, seed=1)
    out = torch.empty(1, T, H, 128, dtype=torch.bfloat16)
    torch_ref(d['q'], d['k'], d['v'], d['g'], d['beta'], d['scale'], out,
              d['A_log'], d['dt_bias'], d['lower_bound'])
    naive, _ = naive_kda(d['q'][0], d['k'][0], d['v'][0], d['g'][0], d['beta'][0],
                         d['scale'], d['A_log'], d['dt_bias'], d['lower_bound'])
    r = rel_rms(out[0], naive)
    print(f"[3] tail chunk            T={T} ({T % 16} rows in last)  "
          f"rel RMS = {r:.2e}  {'PASS' if r < TOL else 'FAIL'}")
    return r < TOL


def check_state(T=32, H=1, D=128):
    """initial_state / final_state, and the layout convention at the boundary.

    torch_ref stores state as [value, key] and uses .t() on it, so the naive
    [key, value] state has to be transposed to compare. Asserting that here is
    what pins the convention the Ascend kernel must match.
    """
    from torch_ref import torch_ref
    d = make(T, H, D, seed=2)
    torch.manual_seed(7)
    s0_kv = torch.randn(1, H, D, D, dtype=torch.float64) * 0.05   # [key, value]
    s0_vk = s0_kv.transpose(-1, -2).contiguous()                  # [value, key]

    init = s0_vk.to(torch.bfloat16)
    final = torch.empty(1, H, D, D, dtype=torch.bfloat16)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16)
    torch_ref(d['q'], d['k'], d['v'], d['g'], d['beta'], d['scale'], out,
              d['A_log'], d['dt_bias'], d['lower_bound'],
              initial_state=init, final_state=final)

    naive, naive_final = naive_kda(
        d['q'][0], d['k'][0], d['v'][0], d['g'][0], d['beta'][0], d['scale'],
        d['A_log'], d['dt_bias'], d['lower_bound'], state0=s0_kv)

    r_out = rel_rms(out[0], naive)
    # final_state comes back as [value, key]; naive holds [key, value].
    r_state = rel_rms(final[0].double().transpose(-1, -2), naive_final[0])
    ok = r_out < TOL and r_state < TOL
    print(f"[4] with state            out rel RMS = {r_out:.2e}   "
          f"final_state (transposed) rel RMS = {r_state:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if ok:
        print("    -> confirms GM state is [value, key]; the Ascend kernel "
              "carries [key, value] and owes a boundary transpose")
    return ok


def check_varlen(H=1, D=128):
    from torch_ref import torch_ref
    lens = [16, 24, 8]          # deliberately ragged, none a multiple of 16 twice
    T = sum(lens)
    d = make(T, H, D, seed=3)
    bounds = [0]
    for L in lens:
        bounds.append(bounds[-1] + L)
    cu = torch.tensor(bounds, dtype=torch.long)

    out = torch.empty(1, T, H, D, dtype=torch.bfloat16)
    torch_ref(d['q'], d['k'], d['v'], d['g'], d['beta'], d['scale'], out,
              d['A_log'], d['dt_bias'], d['lower_bound'], cu_seqlens=cu)

    seg = list(zip(bounds[:-1], bounds[1:]))
    naive, _ = naive_kda(d['q'][0], d['k'][0], d['v'][0], d['g'][0], d['beta'][0],
                         d['scale'], d['A_log'], d['dt_bias'], d['lower_bound'],
                         seg=seg)
    r = rel_rms(out[0], naive)
    print(f"[5] varlen {lens}       rel RMS = {r:.2e}  "
          f"{'PASS' if r < TOL else 'FAIL'}")
    return r < TOL


if __name__ == "__main__":
    results = []
    for fn in (check_neumann, check_basic, check_tail, check_state, check_varlen):
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"    {fn.__name__} raised {type(exc).__name__}: {exc}")
            results.append(False)
    print()
    print(f"{sum(results)}/{len(results)} checks passed "
          f"(agreement floor for bf16 is ~1e-2)")
    sys.exit(0 if all(results) else 1)
