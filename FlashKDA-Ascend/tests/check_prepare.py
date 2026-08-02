"""Check kernel1's prepare phase against CPU, field by field.

Kernel1's two AIV phases already run correctly on the card; only the cube phases
hang. That means the prepare output -- k_decayed, q_decayed, k_inv, k_restored
and g_total -- can be verified numerically right now, without the GEMMs working.

This is worth doing before chasing the cube any further: it confirms on real
hardware that the L2 normalization, the gate activation, the cumulative sum and
all four decay terms are right, which is the bulk of the algorithm and the part
where the inherited draft had the most numerical bugs (the log2(e) double-count,
the aliased inv_cumsum, the two-phase UB overlap).

Run with a card:
    PYTHONPATH=. python3 tests/check_prepare.py

Without a card it still self-tests: --cpu-only checks the expected values
against torch_ref's own intermediates, so the expectations themselves are not
just a second copy of the same possible mistake.
"""
import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = 128


def gate_and_decay(q, k, g, beta, A_log, dt_bias, scale, lower_bound):
    """CPU expectation for one [CHUNK, D] chunk of one head, in float64.

    Mirrors what kernel1's RunPrepare should leave in the workspace. Written
    from the recurrence rather than copied from torch_ref, so a shared mistake
    would have to be made twice.
    """
    q = q.double()
    k = k.double()
    g = g.double()

    # L2 normalize per row
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # gate: gv = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    a = math.exp(float(A_log))
    gv = lower_bound * torch.sigmoid(a * (g + dt_bias.double()))

    gc = gv.cumsum(dim=0)              # inclusive cumsum down the 16 rows
    g_total = torch.exp(gc[-1])        # already exponentiated, as stored

    pos = torch.exp(gc)
    neg = torch.exp(-gc)

    return {
        'k_decayed': k * pos,
        'q_decayed': q * pos * scale,
        'k_inv': k * neg,
        'k_restored': k * neg * g_total,
        'g_total': g_total,
    }


def cube_expect(q, k, g, beta, A_log, dt_bias, scale, lower_bound):
    """CPU expectation for kernel1's AIC outputs: Mqk and INV.

    L    = k_decayed @ k_inv^T, strictly-lower masked and scaled by
           sigmoid(beta[i]) on row i.
    Mqk  = q_decayed @ k_inv^T, lower-triangular including the diagonal.
    INV  = (I + L)^-1, which the kernel builds as
           (I-L)(I+L^2)(I+L^4)(I+L^8).
    """
    d = gate_and_decay(q, k, g, beta, A_log, dt_bias, scale, lower_bound)
    k_dec = d['k_decayed']
    q_dec = d['q_decayed']
    k_inv = d['k_inv']

    L = k_dec @ k_inv.t()
    Mqk = torch.tril(q_dec @ k_inv.t())

    bs = torch.sigmoid(beta.double()).unsqueeze(-1)
    L = torch.tril(L, diagonal=-1) * bs

    I = torch.eye(CHUNK, dtype=torch.float64)
    INV = torch.linalg.inv(I + L)
    return {'Mqk': Mqk, 'INV': INV}


def cpu_selfcheck():
    """Cross-check the expectations above against torch_ref's own intermediates."""
    import torch_ref as TR

    torch.manual_seed(0)
    H = 1
    q = torch.randn(CHUNK, H, D, dtype=torch.bfloat16)
    k = torch.randn(CHUNK, H, D, dtype=torch.bfloat16)
    g = torch.randn(CHUNK, H, D, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, D, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0

    exp = gate_and_decay(q[:, 0], k[:, 0], g[:, 0], None,
                         A_log[0], dt_bias[0], scale, lower_bound)

    # torch_ref's own path, for the same inputs.
    qn = TR.l2_normalize_kernel_match(q)
    kn = TR.l2_normalize_kernel_match(k)
    gg = g.to(torch.float32) + dt_bias.unsqueeze(0)
    a_log_exp = TR.fp32_ex2_ftz(A_log * TR.LOG2E).unsqueeze(0).unsqueeze(-1)
    gg = (lower_bound * TR.LOG2E) * TR.sigmoid_tanh_approx(a_log_exp * gg)
    gc = gg[:, 0, :].cumsum(dim=0)
    ref = {
        'k_decayed': (kn[:, 0] * TR.fp32_ex2_ftz(gc).to(torch.bfloat16)).double(),
        'q_decayed': (qn[:, 0] * TR.fp32_ex2_ftz(gc).to(torch.bfloat16)
                      * torch.tensor(scale, dtype=torch.bfloat16)).double(),
        'k_inv': (kn[:, 0] * TR.fp32_ex2_ftz(-gc).to(torch.bfloat16)).double(),
        'g_total': TR.fp32_ex2_ftz(gc[-1:]).squeeze(0).double(),
    }

    print("cross-checking expectations against torch_ref's intermediates:")
    worst = 0.0
    for name, r in ref.items():
        e = exp[name]
        rel = ((e - r).flatten().square().mean().sqrt() /
               (r.flatten().square().mean().sqrt() + 1e-30)).item()
        worst = max(worst, rel)
        print(f"  {name:<12} rel RMS = {rel:.3e}")
    ok = worst < 5e-2
    print(f"-> {'PASS' if ok else 'FAIL'} (bf16 intermediates put the floor near 1e-2)")
    return ok


def npu_check():
    import torch_npu  # noqa: F401

# Chunk size comes from the extension, not a local constant: a test that
# hardcodes it silently stops matching the kernel when the kernel changes.
from flash_kda import _C as _kda_C
CHUNK = _kda_C.CHUNK
    from flash_kda import fwd, _C

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

    ws_bytes = _C.get_workspace_size(T, H, 1)
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8).to(dev)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=dev)

    _C.fwd(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), scale,
           out, workspace, A_log.to(dev), dt_bias.to(dev), lower_bound)
    torch_npu.npu.synchronize()

    raw = workspace.cpu().numpy().tobytes()
    import struct

    def field(off, count, fmt, size):
        if fmt == 'e':
            # bfloat16, NOT IEEE fp16. struct has no bf16 code and 'e' silently
            # reinterprets the exponent field, which turns correct data into
            # garbage. bf16 is the top 16 bits of the fp32 pattern.
            bits = struct.unpack_from(f'<{count}H', raw, off)
            as_f32 = struct.unpack(f'<{count}f',
                                   struct.pack(f'<{count}I', *(b << 16 for b in bits)))
            return torch.tensor(as_f32)
        return torch.tensor(struct.unpack_from(f'<{count}{fmt}', raw, off))

    # Offsets must match WorkspaceOffsets in layout.hpp.
    off_kdec, off_qdec, off_kinv, off_kres = 0, 4096, 8192, 12288
    off_gtot = 16384

    got = {
        'k_decayed': field(off_kdec, CHUNK * D, 'e', 2).reshape(CHUNK, D),
        'q_decayed': field(off_qdec, CHUNK * D, 'e', 2).reshape(CHUNK, D),
        'k_inv': field(off_kinv, CHUNK * D, 'e', 2).reshape(CHUNK, D),
        'k_restored': field(off_kres, CHUNK * D, 'e', 2).reshape(CHUNK, D),
        'g_total': field(off_gtot, D, 'f', 4),
    }
    exp = gate_and_decay(q[0, :, 0], k[0, :, 0], g[0, :, 0], None,
                         A_log[0], dt_bias[0], scale, lower_bound)

    # Cube outputs, at offsets read from the extension rather than copied --
    # they have moved twice since these were written.
    got['INV'] = field(_kda_C.WS_OFF_KINV, CHUNK * CHUNK, 'e', 2).reshape(CHUNK, CHUNK)
    got['Mqk'] = field(_kda_C.WS_OFF_KMQK, CHUNK * CHUNK, 'e', 2).reshape(CHUNK, CHUNK)
    exp.update(cube_expect(q[0, :, 0], k[0, :, 0], g[0, :, 0], beta[0, :, 0],
                           A_log[0], dt_bias[0], scale, lower_bound))

    print("kernel1 prepare output vs CPU:")
    allok = True
    for name in ('k_decayed', 'q_decayed', 'k_inv', 'k_restored', 'g_total',
                 'Mqk', 'INV'):
        e = exp[name].double()
        a = got[name].double()
        rel = ((a - e).flatten().square().mean().sqrt() /
               (e.flatten().square().mean().sqrt() + 1e-30)).item()
        good = rel < 5e-2
        allok &= good
        print(f"  {name:<12} rel RMS = {rel:.3e}  {'PASS' if good else 'FAIL'}")
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu-only', action='store_true',
                    help='only cross-check the expectations, no device needed')
    args = ap.parse_args()

    if args.cpu_only:
        sys.exit(0 if cpu_selfcheck() else 1)

    ok = cpu_selfcheck()
    print()
    ok &= npu_check()
    sys.exit(0 if ok else 1)
