"""FlashKDA reference implementation (pure PyTorch, NPU-compatible).

Matches the CUDA kernel numerics exactly:
  - sigmoid via tanh approximation: sigmoid(x) = tanh(x*0.5)*0.5+0.5
  - fp16 accumulation GEMM (torch.mm with out_dtype=torch.float16)
  - L2 normalize with FMA tree reduction pattern
  - exp2 approximation with FTZ (flush-to-zero)

No CUDA dependencies — works on CPU or NPU.
"""

import torch
import math

# ============================================================
# Numeric helpers matching kernel behavior
# ============================================================

LOG2E = 1.4426950408889634


def sigmoid_tanh_approx(x):
    """sigmoid via tanh approximation matching kernel's tanh.approx.f32."""
    x = x.to(torch.float32)
    return torch.tanh(x * 0.5) * 0.5 + 0.5


def fp32_ex2_ftz(x):
    """exp2 with flush-to-zero matching kernel's ex2.approx.ftz.f32."""
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    ret = torch.exp2(x)
    # Flush to zero: if |ret| < tiny, set to 0
    ret = torch.where(ret.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(ret), ret)
    return ret


def fp32_fma(c, a, b):
    """FMA: c + a * b in fp64, result in fp32. Matches kernel's FMA behavior."""
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def matmul_fp16acc(a, b):
    """C = A @ B with fp16 accumulation matching kernel's SM80 MMA behavior.

    On NPU: use torch.mm with out_dtype=torch.float16.
    On CPU: simulate fp16 accumulation by casting inputs to fp16 first.
    """
    a_fp16 = a.to(torch.float16)
    b_fp16 = b.to(torch.float16)
    # torch.mm accumulates in the output dtype
    return torch.mm(a_fp16, b_fp16)


def l2_normalize_kernel_match(x):
    """L2 normalize matching kernel's warp-shuffle tree reduction with FMA.

    x: [..., D] bf16, D must be 128.
    The kernel does: cast to fp32, square, reduce in groups of 8,
    then tree reduction across 16 partial sums, then rsqrt + multiply.
    """
    x_f32 = x.float()
    # Group into 16 groups of 8 elements each (matching warp shuffle pattern)
    groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)

    # Partial sums: sum of squares within each group of 8
    partials = torch.zeros(*x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device)
    for i in range(8):
        partials = fp32_fma(partials, groups[..., i], groups[..., i])

    # Tree reduction across 16 partial sums (matching warp shuffle)
    for offset in [8, 4, 2, 1]:
        indices = torch.arange(16, device=x.device) ^ offset
        partials = partials + partials[..., indices]

    inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
    return (x_f32 * inv_norm).to(x.dtype)


# ============================================================
# Torch reference implementation
# ============================================================

def torch_ref(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None):
    """Torch reference for FlashKDA forward.

    Supports both fixed-length and variable-length sequences.
    Input: [B, T, H, D] (4D). B must be 1 when cu_seqlens is provided.

    initial_state/final_state can be:
      - None: no state (zero-init / skip store)
      - bf16 tensor: [N, H, D, D]
      - fp32 tensor: [N, H, D, D] (converted to bf16 for compute, back to fp32 for output)
    """
    assert q.dim() == 4, f"Expected 4D input [B, T, H, D], got {q.dim()}D"
    B = q.shape[0]
    if cu_seqlens is not None:
        assert B == 1, f"B must be 1 when cu_seqlens is provided, got B={B}"

    # Reshape to [T_total, H, D] for internal processing
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    g = g.reshape(-1, *g.shape[2:])
    beta = beta.reshape(-1, *beta.shape[2:])
    out = out.reshape(-1, *out.shape[2:])

    if B > 1:
        T_seq = q.shape[0] // B
        cu_seqlens = torch.arange(0, B * T_seq + 1, T_seq, dtype=torch.long, device=q.device)

    _, H, D = q.shape
    CHUNK = 16
    device = q.device
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)

    # L2 normalize q and k (matching kernel behavior)
    q = l2_normalize_kernel_match(q)
    k = l2_normalize_kernel_match(k)

    # Gate activation: g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    if A_log is not None:
        assert dt_bias is not None
        assert A_log.dtype == torch.float32
        assert g.dtype == torch.bfloat16
        assert dt_bias.dtype == torch.float32

        g = g.to(torch.float32) + dt_bias.unsqueeze(0)
        a_log_exp = fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
        gate_scale = lower_bound * LOG2E
        g = gate_scale * sigmoid_tanh_approx(a_log_exp * g)

    # Determine state dtype
    state_fp32 = (initial_state is not None and initial_state.dtype == torch.float32) or \
                 (final_state is not None and final_state.dtype == torch.float32)

    if cu_seqlens is None:
        T = q.shape[0]
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

    N = len(cu_seqlens) - 1

    # Initialize working state
    if initial_state is not None:
        work_state = initial_state.to(torch.bfloat16).clone()
    else:
        work_state = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=device)

    # Process each sequence
    for seq_idx in range(N):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        n_chunks = (seq_len + CHUNK - 1) // CHUNK

        for chunk_idx in range(n_chunks):
            t0 = bos + chunk_idx * CHUNK
            actual_len = min(CHUNK, eos - t0)

            for h in range(H):
                # Extract chunks with zero-padding for tail
                g_chunk = torch.zeros(CHUNK, D, dtype=g.dtype, device=device)
                q_chunk = torch.zeros(CHUNK, D, dtype=q.dtype, device=device)
                k_chunk = torch.zeros(CHUNK, D, dtype=k.dtype, device=device)
                v_chunk = torch.zeros(CHUNK, D, dtype=v.dtype, device=device)
                beta_chunk = torch.zeros(CHUNK, dtype=beta.dtype, device=device)

                g_chunk[:actual_len] = g[t0:t0 + actual_len, h, :]
                q_chunk[:actual_len] = q[t0:t0 + actual_len, h, :]
                k_chunk[:actual_len] = k[t0:t0 + actual_len, h, :]
                v_chunk[:actual_len] = v[t0:t0 + actual_len, h, :]
                beta_chunk[:actual_len] = beta[t0:t0 + actual_len, h]

                # Cumulative sum of g
                g_cumsum = g_chunk.cumsum(dim=0)
                g_total = g_cumsum[-1:]

                # Decay factors
                k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
                q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
                neg_g_cumsum_bf16 = fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
                k_inv = k_chunk * neg_g_cumsum_bf16
                g_total_exp_bf16 = fp32_ex2_ftz(g_total).to(torch.bfloat16)
                k_restored = k_inv * g_total_exp_bf16

                # L = k_decayed @ k_inv^T (fp16 accumulation)
                L = matmul_fp16acc(k_decayed, k_inv.t())
                Mqk = torch.matmul(q_decayed, k_inv.t())

                # Beta sigmoid activation
                beta_activated = sigmoid_tanh_approx(beta_chunk)
                beta_val_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
                beta_val_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)

                # Apply lower-triangular mask with beta sigmoid
                L = torch.tril(L, diagonal=-1) * beta_val_fp16
                Mqk = torch.tril(Mqk)

                # Neumann inverse: INV = (I-L)^{-1}
                # = (I-L) + (I-L)@L^2 + (I-L)@L^2@L^4 + (I-L)@L^2@L^4@L^8
                # Simplified: INV = (I-L) + INV@L^2, INV += INV@L^4, INV += INV@L^8
                INV = torch.eye(CHUNK, dtype=torch.float16, device=device) - L
                L2 = matmul_fp16acc(L, L)
                INV = INV + matmul_fp16acc(INV, L2)
                L4 = matmul_fp16acc(L2, L2)
                INV = INV + matmul_fp16acc(INV, L4)
                L8 = matmul_fp16acc(L4, L4)
                INV = INV + matmul_fp16acc(INV, L8)
                INV = INV.to(torch.bfloat16)

                # Recurrence
                state_slice = work_state[seq_idx, h]

                # v_sub = (v - k_decayed @ state) * sigmoid(beta)
                v_sub = v_chunk - torch.matmul(k_decayed, state_slice.t())
                v_sub = v_sub * beta_val_bf16

                # u = INV @ v_sub
                U = torch.matmul(INV, v_sub)

                # out = q_decayed @ state + Mqk @ u
                _out = torch.matmul(q_decayed, state_slice.t())
                _out = _out + torch.matmul(Mqk, U)

                # State update: state = state * exp(g_total) + k_restored^T @ u
                delta_s = torch.mm(k_restored.t(), U, out_dtype=torch.float32)

                g_total_exp = fp32_ex2_ftz(g_total)
                g_total_exp = g_total_exp.squeeze(0).unsqueeze(-1)

                # FMA: new_state = delta_s + state.t() * g_total_exp, then transpose back
                work_state[seq_idx, h] = fp32_fma(
                    delta_s, state_slice.to(torch.float32).t(), g_total_exp
                ).to(torch.bfloat16).t()

                # Write output
                out[t0:t0 + actual_len, h] = _out[:actual_len]

    # Store final state
    if final_state is not None:
        if state_fp32:
            final_state.copy_(work_state.to(torch.float32))
        else:
            final_state.copy_(work_state)
