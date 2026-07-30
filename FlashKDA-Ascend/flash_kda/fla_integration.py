"""FlashKDA-Ascend FLA (flash-linear-attention) integration.

When flash_kda is installed alongside flash-linear-attention (FLA),
the FLA chunk_kda operation automatically dispatches to flash_kda
for improved performance on Ascend NPU.

Usage:
    import flash_kda.fla_integration  # registers the dispatch
    from fla.ops.kda import chunk_kda  # now uses flash_kda on NPU

Opt out:
    FLA_FLASH_KDA=0 python your_script.py

Debug:
    import logging
    logging.basicConfig(level=logging.INFO)
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)

# Check if FLA integration is disabled
_FLASH_KDA_ENABLED = os.environ.get("FLA_FLASH_KDA", "1") != "0"


def _flash_kda_fla_fwd(
    q, k, v, g, beta,
    scale=1.0,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    use_gate_in_kernel=True,
    use_qk_l2norm_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    transpose_state_layout=True,
    head_first=False,
):
    """FLA-compatible wrapper for flash_kda.fwd().

    Translates FLA's chunk_kda interface to flash_kda's fwd() interface.
    """
    from flash_kda import fwd

    if head_first:
        # FLA head-first layout: [B, H, T, D] -> [B, T, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous() if g is not None else None
        beta = beta.transpose(1, 2).contiguous() if beta is not None else None

    B, T, H, D = q.shape
    assert D == 128, f"flash_kda requires D=128, got D={D}"

    # Allocate output
    out = torch.empty_like(q)

    # Handle state
    final_state = None
    if output_final_state:
        N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
        state_dtype = initial_state.dtype if initial_state is not None else q.dtype
        final_state = torch.empty(N, H, D, D, device=q.device, dtype=state_dtype)

    # Default A_log and dt_bias if not provided
    if A_log is None:
        A_log = torch.zeros(H, device=q.device, dtype=torch.float32)
    if dt_bias is None:
        dt_bias = torch.zeros(H, D, device=q.device, dtype=torch.float32)
    if lower_bound is None:
        lower_bound = -5.0

    # Call flash_kda
    with torch.inference_mode():
        fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
            initial_state=initial_state, final_state=final_state,
            cu_seqlens=cu_seqlens)

    if head_first:
        out = out.transpose(1, 2).contiguous()

    if output_final_state:
        return out, final_state
    return out


def install_fla_dispatch():
    """Install flash_kda as the backend for fla.ops.kda.chunk_kda.

    This monkey-patches FLA's chunk_kda to use flash_kda when running
    on Ascend NPU. The original implementation is preserved for CPU/CUDA.
    """
    if not _FLASH_KDA_ENABLED:
        logger.info("FLA_FLASH_KDA=0: flash_kda dispatch disabled")
        return

    try:
        from fla.ops.kda import chunk_kda
        import fla.ops.kda as kda_module

        # Save original
        _original_chunk_kda = kda_module.chunk_kda

        def _dispatched_chunk_kda(*args, **kwargs):
            # Check if inputs are on NPU
            q = args[0] if args else kwargs.get('q')
            if q is not None and q.is_npu():
                logger.debug("flash_kda: dispatching chunk_kda to Ascend NPU backend")
                return _flash_kda_fla_fwd(*args, **kwargs)
            else:
                return _original_chunk_kda(*args, **kwargs)

        kda_module.chunk_kda = _dispatched_chunk_kda
        logger.info("flash_kda: registered FLA dispatch for chunk_kda on NPU")

    except ImportError:
        logger.debug("flash_kda: FLA not installed, skipping dispatch registration")
    except Exception as e:
        logger.warning(f"flash_kda: failed to register FLA dispatch: {e}")


# Auto-register on import
install_fla_dispatch()
