"""FlashKDA: Flash Kimi Delta Attention (Ascend NPU)"""

import torch
try:  # registers the npu device; absent on a CPU-only host
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None
from flash_kda._C import fwd as _fwd_raw, get_workspace_size


# Reused across calls, allocated on device, and deliberately not zeroed.
#
# This used to be `torch.zeros(...).to(device)` on every call. At T=1024 H=8
# that is 302 MB of host-side zeros copied to the device per call, and it
# measured as 88-98% of end-to-end time -- the kernel itself is 2.81 ms, the
# wrapper made it 29.96 ms.
#
# The zeroing was there in case some slot were read before being written, which
# would make results depend on whatever was in device memory. tests/
# workspace_poison.py checks that directly: it runs with the workspace filled
# with 0xFF and with 0x5A and compares against the zeroed run. 0xFF is NaN as
# bf16, so anything reading uninitialized workspace propagates NaN into the
# output rather than a small perturbation that might pass unnoticed. Output is
# bit-identical at T=64 H=4, 256 H=8 and 1024 H=8, so nothing is read before it
# is written and neither the zeroing nor the host copy is needed.
#
# Not thread-safe: concurrent calls on one device would share the buffer. Call
# clear_workspace_cache() to release it.
_WS_CACHE = {}


def _workspace(nbytes, device):
    key = (device.type, device.index)
    buf = _WS_CACHE.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _WS_CACHE[key] = buf
    return buf[:nbytes]


def clear_workspace_cache():
    """Release cached workspaces. Call between very different shapes to give
    the memory back, or before measuring peak memory."""
    _WS_CACHE.clear()


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None):
    """FlashKDA forward (Flash Kimi Delta Attention) on Ascend NPU.

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be NPU, contiguous, and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    workspace = _workspace(get_workspace_size(T_total, H, N), q.device)

    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)
