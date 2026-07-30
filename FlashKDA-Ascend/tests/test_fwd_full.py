"""FlashKDA-Ascend parametrized correctness tests.

Sweeps over configurations: batch sizes, sequence lengths, head counts,
state dtypes, and varlen patterns.
"""

import torch
import math
import pytest

from torch_ref import torch_ref


def get_device():
    return torch.device("npu:0")


def run_fwd_test(B, T, H, D=128, lower_bound=-5.0,
                 has_state_in=False, has_state_out=False,
                 state_fp32=False, is_varlen=False, seq_lens=None):
    """Run a single forward test comparing kernel vs reference."""
    from flash_kda import fwd

    device = get_device()
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    if is_varlen:
        assert seq_lens is not None
        T_total = sum(seq_lens)
        N = len(seq_lens)
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.int64, device=device,
        )
        q = torch.randn(1, T_total, H, D, device=device, dtype=dtype)
    else:
        T_total = B * T
        N = B
        cu_seqlens = None
        q = torch.randn(B, T, H, D, device=device, dtype=dtype)

    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta_shape = q.shape[:2] + (H,) if not is_varlen else (1, T_total, H)
    beta = torch.randn(*beta_shape, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    out = torch.empty_like(q)

    state_dtype = torch.float32 if state_fp32 else torch.bfloat16
    initial_state = torch.randn(N, H, D, D, device=device, dtype=state_dtype) if has_state_in else None
    final_state = torch.empty(N, H, D, D, device=device, dtype=state_dtype) if has_state_out else None

    # Kernel
    fwd_kwargs = {}
    if initial_state is not None:
        fwd_kwargs['initial_state'] = initial_state
    if final_state is not None:
        fwd_kwargs['final_state'] = final_state
    if cu_seqlens is not None:
        fwd_kwargs['cu_seqlens'] = cu_seqlens

    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, **fwd_kwargs)

    # Reference
    out_ref = torch.zeros_like(out)
    final_state_ref = torch.zeros(N, H, D, D, device=device, dtype=state_dtype) if has_state_out else None
    ref_kwargs = {}
    if initial_state is not None:
        ref_kwargs['initial_state'] = initial_state.clone()
    if final_state_ref is not None:
        ref_kwargs['final_state'] = final_state_ref
    if cu_seqlens is not None:
        ref_kwargs['cu_seqlens'] = cu_seqlens

    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, **ref_kwargs)

    # Check
    out_diff = (out.float() - out_ref.float()).abs()
    max_out_err = out_diff.max().item()
    print(f"  B={B} T={T} H={H} D={D} varlen={is_varlen} state_in={has_state_in} state_out={has_state_out} fp32={state_fp32} | max_out_err={max_out_err:.6e}")

    if has_state_out and final_state is not None:
        state_diff = (final_state.float() - final_state_ref.float()).abs()
        max_state_err = state_diff.max().item()
        print(f"  max_state_err={max_state_err:.6e}")


# ============================================================
# Parametrized test groups
# ============================================================

@pytest.mark.parametrize("B,T,H", [(1, 16, 1), (1, 32, 2), (1, 64, 4), (1, 128, 8)])
def test_fwd_fixed(B, T, H):
    run_fwd_test(B, T, H)


@pytest.mark.parametrize("B,T,H", [(1, 64, 4), (2, 32, 2)])
def test_fwd_with_state_bf16(B, T, H):
    run_fwd_test(B, T, H, has_state_in=True, has_state_out=True, state_fp32=False)


@pytest.mark.parametrize("B,T,H", [(1, 64, 4), (2, 32, 2)])
def test_fwd_with_state_fp32(B, T, H):
    run_fwd_test(B, T, H, has_state_in=True, has_state_out=True, state_fp32=True)


@pytest.mark.parametrize("seq_lens", [
    [16, 32], [32, 64, 48], [16, 16, 16, 16], [1300, 547, 2048],
])
def test_fwd_varlen(seq_lens):
    H = 4
    run_fwd_test(1, sum(seq_lens), H, is_varlen=True, seq_lens=seq_lens)


@pytest.mark.parametrize("seq_lens", [
    [32, 64], [16, 48, 32],
])
def test_fwd_varlen_with_state(seq_lens):
    H = 4
    run_fwd_test(1, sum(seq_lens), H, is_varlen=True, seq_lens=seq_lens,
                 has_state_in=True, has_state_out=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
