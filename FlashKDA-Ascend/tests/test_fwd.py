"""FlashKDA-Ascend correctness tests.

Compares flash_kda.fwd() output against the PyTorch reference implementation.
"""

import torch
import torch_npu  # registers the npu device with torch
import math
import pytest

from torch_ref import torch_ref


# ============================================================
# Test helpers
# ============================================================

def get_device():
    return torch.device("npu:0")


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def print_error_stats(label, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    ref_abs = expected.float().abs()
    avg_rtol = diff.mean() / (ref_abs.mean() + 1e-8)
    max_rtol = diff.max() / (ref_abs.max() + 1e-8)
    print(f"  {label} | avg_rtol: {avg_rtol:.6e}, max_rtol: {max_rtol:.6e}")
    print(f"  {label} | avg_atol: {diff.mean():.6e}, max_atol: {diff.max():.6e}")


def make_inputs(B, T, H, D=128, device=None, dtype=torch.bfloat16):
    """Create random test inputs matching FlashKDA shapes."""
    if device is None:
        device = get_device()
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    out = torch.empty(B, T, H, D, device=device, dtype=dtype)
    return q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound


# ============================================================
# Tests
# ============================================================

def test_fwd_basic():
    """Basic forward: B=1, T=64, H=4, D=128 vs torch_ref."""
    from flash_kda import fwd

    B, T, H, D = 1, 64, 4, 128
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound = make_inputs(B, T, H)

    # Kernel output
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)

    # Reference output
    out_ref = torch.zeros_like(out)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound)

    print_error_stats("output", out, out_ref)
    assert out.shape == (B, T, H, D)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_fwd_with_state():
    """Forward with initial and final state (bf16)."""
    from flash_kda import fwd

    B, T, H, D = 1, 64, 4, 128
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound = make_inputs(B, T, H)
    initial_state = torch.randn(B, H, D, D, device=out.device, dtype=torch.bfloat16)
    final_state = torch.empty(B, H, D, D, device=out.device, dtype=torch.bfloat16)

    # Kernel
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=initial_state, final_state=final_state)

    # Reference
    out_ref = torch.zeros_like(out)
    final_state_ref = torch.zeros_like(initial_state)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound,
              initial_state=initial_state.clone(), final_state=final_state_ref)

    print_error_stats("output", out, out_ref)
    print_error_stats("final_state", final_state, final_state_ref)


def test_fwd_fp32_state():
    """Forward with fp32 state."""
    from flash_kda import fwd

    B, T, H, D = 1, 64, 4, 128
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound = make_inputs(B, T, H)
    initial_state = torch.randn(B, H, D, D, device=out.device, dtype=torch.float32)
    final_state = torch.empty(B, H, D, D, device=out.device, dtype=torch.float32)

    # Kernel
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=initial_state, final_state=final_state)

    # Reference
    out_ref = torch.zeros_like(out)
    final_state_ref = torch.zeros_like(initial_state)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound,
              initial_state=initial_state.clone(), final_state=final_state_ref)

    print_error_stats("output", out, out_ref)
    print_error_stats("final_state", final_state, final_state_ref)


def test_fwd_varlen():
    """Forward with variable-length sequences (cu_seqlens)."""
    from flash_kda import fwd

    H, D = 4, 128
    seq_lens = [32, 64, 48, 16]
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.int64, device=get_device(),
    )

    q = torch.randn(1, T_total, H, D, device=get_device(), dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, T_total, H, device=get_device(), dtype=torch.bfloat16)
    A_log = torch.randn(H, device=get_device(), dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=get_device(), dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    out = torch.empty_like(q)

    # Kernel
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, cu_seqlens=cu_seqlens)

    # Reference
    out_ref = torch.zeros_like(out)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, cu_seqlens=cu_seqlens)

    print_error_stats("output", out, out_ref)
    assert out.shape == (1, T_total, H, D)


def test_fwd_varlen_with_state():
    """Varlen forward with initial and final state."""
    from flash_kda import fwd

    H, D = 4, 128
    seq_lens = [32, 64, 48]
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.int64, device=get_device(),
    )

    q = torch.randn(1, T_total, H, D, device=get_device(), dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, T_total, H, device=get_device(), dtype=torch.bfloat16)
    A_log = torch.randn(H, device=get_device(), dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=get_device(), dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)
    lower_bound = -5.0
    out = torch.empty_like(q)
    initial_state = torch.randn(N, H, D, D, device=get_device(), dtype=torch.bfloat16)
    final_state = torch.empty(N, H, D, D, device=get_device(), dtype=torch.bfloat16)

    # Kernel
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)

    # Reference
    out_ref = torch.zeros_like(out)
    final_state_ref = torch.zeros_like(initial_state)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound,
              initial_state=initial_state.clone(), final_state=final_state_ref,
              cu_seqlens=cu_seqlens)

    print_error_stats("output", out, out_ref)
    print_error_stats("final_state", final_state, final_state_ref)


def test_fwd_large():
    """Large sequence: B=1, T=2048, H=8, D=128."""
    from flash_kda import fwd

    B, T, H, D = 1, 2048, 8, 128
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound = make_inputs(B, T, H)

    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)

    out_ref = torch.zeros_like(out)
    torch_ref(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound)

    print_error_stats("output", out, out_ref)
    assert not torch.isnan(out).any(), "Output contains NaN"


if __name__ == "__main__":
    test_fwd_basic()
    print("test_fwd_basic PASSED")
    test_fwd_with_state()
    print("test_fwd_with_state PASSED")
    test_fwd_fp32_state()
    print("test_fwd_fp32_state PASSED")
    test_fwd_varlen()
    print("test_fwd_varlen PASSED")
    test_fwd_varlen_with_state()
    print("test_fwd_varlen_with_state PASSED")
    test_fwd_large()
    print("test_fwd_large PASSED")
    print("All tests PASSED")
