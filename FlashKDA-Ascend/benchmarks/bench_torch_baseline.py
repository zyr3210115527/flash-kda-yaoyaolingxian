"""PyTorch baseline for KDA forward, on the same NPU as the kernel.

STATUS: cannot run on the current devspace. Every aclnn operator fails with
error 561103 ("Failed to ParseDynamicKernels") -- not just exp and cast, but
add and matmul too. The cause is that this image installs the CANN toolkit
without the binary operator package: the kernel directory

    $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend910b/

holds only ops_oam, with none of the precompiled TBE kernels torch_npu
dispatches to. Our own kernels are unaffected because .asc files are compiled
directly by the Ascend compiler and never go through that path, which is why
the correctness suite (whose reference runs on CPU) works fine here.

To get this number, run on a devspace whose image includes
Ascend-cann-kernels-910b, or install that package alongside the toolkit. The
script is left ready to run as-is.


The CUDA number in BENCHMARK.md is measured on an H20, so it says as much about
the hardware as about the code. This runs a straightforward PyTorch
implementation on the same 910B card, which answers the question the kernel
actually has to justify: is a hand-written Ascend C kernel faster than just
writing it in torch on this device?

Two baselines, because they measure different things:

  torch_ref     the bit-exact emulator in tests/torch_ref.py. It loops over
                chunks in Python and reproduces the kernel's rounding step by
                step, so it is a correctness oracle, not an implementation
                anyone would ship. Included because it is the only thing
                guaranteed to match, and it bounds how slow naive looks.

  torch_chunked the implementation someone would actually write: the intra-chunk
                work (normalize, decay, the L matrix, its inverse) batched
                across all chunks and heads at once, with a Python loop only
                over the inter-chunk state recurrence, which is genuinely
                sequential. This is the number the kernel has to beat.

Both run in bf16 with fp32 accumulation, matching the kernel.
"""
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
D = 128
CHUNK = 16


def sync():
    torch_npu.npu.synchronize()


def time_it(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def torch_chunked(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound):
    """Vectorized chunked delta-rule forward.

    Shapes: q,k,v,g are [1, T, H, D]; beta is [1, T, H].
    Everything intra-chunk is batched over (num_chunks * H); only the state
    recurrence loops.
    """
    _, T, H, _ = q.shape
    nc = T // CHUNK

    # [H] and [H,D] parameters broadcast over tokens.
    a = -torch.exp(A_log.float())                       # [H]
    gate = g.float() + dt_bias.unsqueeze(0).unsqueeze(0)   # [1,T,H,D]
    gate = torch.nn.functional.softplus(gate) * a.view(1, 1, H, 1)
    gate = gate.clamp(min=lower_bound)

    def to_chunks(x):    # [1,T,H,D] -> [nc*H, CHUNK, D]
        return (x.view(nc, CHUNK, H, D).permute(0, 2, 1, 3).reshape(nc * H, CHUNK, D))

    qc = to_chunks(q.float()) * scale
    kc = to_chunks(k.float())
    vc = to_chunks(v.float())
    gc = to_chunks(gate)
    bc = beta.float().view(nc, CHUNK, H).permute(0, 2, 1).reshape(nc * H, CHUNK)
    bc = torch.sigmoid(bc)

    qc = torch.nn.functional.normalize(qc, dim=-1)
    kc = torch.nn.functional.normalize(kc, dim=-1)

    gcum = gc.cumsum(dim=1)                       # cumulative decay in-chunk
    kd = kc * torch.exp(-gcum)
    qd = qc * torch.exp(gcum)

    kb = kd * bc.unsqueeze(-1)

    # Strictly lower triangular L, then (I + L)^-1 by the same Neumann series
    # the kernel uses, batched over every chunk and head at once.
    L = torch.tril(kb @ kd.transpose(1, 2), diagonal=-1)
    eye = torch.eye(CHUNK, device=q.device, dtype=L.dtype).expand_as(L)
    INV = eye - L
    P = L @ L
    for _ in range(3):
        INV = INV + INV @ P
        P = P @ P

    w = INV @ kb
    u = INV @ (vc * bc.unsqueeze(-1))

    # Sequential across chunks: the state carries forward and cannot batch.
    state = torch.zeros(H, D, D, dtype=torch.float32, device=q.device)
    out = torch.empty(nc * H, CHUNK, D, dtype=torch.float32, device=q.device)
    decay_end = torch.exp(gcum[:, -1, :])         # [nc*H, D]
    for c in range(nc):
        sl = slice(c * H, (c + 1) * H)
        s = state
        o_intra = torch.tril(qd[sl] @ kd[sl].transpose(1, 2), diagonal=-1) @ u[sl]
        out[sl] = qd[sl] @ s + o_intra
        state = s * decay_end[sl].unsqueeze(-1) + kd[sl].transpose(1, 2) @ (u[sl] - w[sl] @ s)

    return out.view(nc, H, CHUNK, D).permute(0, 2, 1, 3).reshape(1, T, H, D).to(torch.bfloat16)


def main():
    from flash_kda import _C

    print(f"{'T':>6} {'H':>4} {'kernel':>10} {'torch_chunked':>14} "
          f"{'torch_ref':>11} {'vs chunked':>11}")

    for T, H in [(256, 4), (512, 8), (1024, 8), (2048, 8)]:
        torch.manual_seed(0)
        mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
        q, k, v, g = (mk(1, T, H, D) for _ in range(4))
        beta = mk(1, T, H)
        A_log = mk(H, dt=torch.float32)
        dt_bias = mk(H, D, dt=torch.float32)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
        scale = 1.0 / math.sqrt(D)
        ws = torch.zeros(_C.get_workspace_size(T, H, 1), dtype=torch.uint8).to(DEV)

        t_kern = time_it(lambda: _C.fwd(q, k, v, g, beta, scale, out, ws,
                                        A_log, dt_bias, -5.0))
        t_ch = time_it(lambda: torch_chunked(q, k, v, g, beta, scale,
                                             A_log, dt_bias, -5.0))

        t_ref = float('nan')
        if T <= 512:      # the Python-loop emulator is far too slow beyond this
            from torch_ref import torch_ref as _ref
            o2 = torch.empty_like(out)
            t_ref = time_it(lambda: _ref(q, k, v, g, beta, scale, o2,
                                         A_log, dt_bias, -5.0),
                            warmup=1, iters=2)

        ratio = t_ch / t_kern
        print(f"{T:>6} {H:>4} {t_kern:>9.2f}m {t_ch:>13.2f}m "
              f"{t_ref:>10.1f}m {ratio:>10.2f}x")

        del ws
        torch_npu.npu.empty_cache()

    print()
    print("'vs chunked' > 1 means the kernel is faster than PyTorch on this card.")


if __name__ == "__main__":
    main()
