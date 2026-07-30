# FlashKDA-Ascend

FlashKDA: Flash Kimi Delta Attention — high-performance KDA attention kernel on Huawei Ascend NPU, built on catlass (CANN Templates for Linear Algebra Subroutines).

Port of [FlashKDA](https://github.com/MoonshotAI/FlashKDA) (NVIDIA CUTLASS) to Ascend NPU architecture. Same mathematical algorithm, completely reimplemented using AscendC AIC+AIV programming model.

## Requirements

- **Ascend NPU**: Atlas A2/A3 (architecture `2201`) or Ascend 950 (`3510`)
- **CANN SDK**: >= 8.5.0 (`ASCEND_HOME_PATH` environment variable must be set)
- **PyTorch** + **torch_npu**
- **catlass**: included as git submodule

## Installation

```bash
git clone https://github.com/YOUR_ORG/FlashKDA-Ascend.git
cd FlashKDA-Ascend
git submodule update --init --recursive   # fetches catlass
pip install -v --no-build-isolation .
```

### Architecture Selection

| Environment Variable | Values | Default |
|---|---|---|
| `FLASH_KDA_ASCEND_ARCH` | `2201` (Atlas A2/A3), `3510` (Ascend 950) | `2201` |
| `ASCEND_HOME_PATH` | Path to CANN SDK | Required |

## Quick Start

```python
import torch
import torch_npu
import math
from flash_kda import fwd

B, T, H, D = 1, 1024, 8, 128
device = torch.device("npu:0")

q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
A_log = torch.randn(H, device=device, dtype=torch.float32)
dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)

scale = 1.0 / math.sqrt(D)
lower_bound = -5.0
out = torch.empty_like(q)

fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
```

### With State

```python
initial_state = torch.randn(B, H, D, D, device=device, dtype=torch.bfloat16)
final_state = torch.empty(B, H, D, D, device=device, dtype=torch.bfloat16)

fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
    initial_state=initial_state, final_state=final_state)
```

### Variable-Length Sequences

```python
cu_seqlens = torch.tensor([0, 256, 512, 1024], dtype=torch.int64, device=device)
q = torch.randn(1, 1024, H, D, device=device, dtype=torch.bfloat16)
# ... other inputs ...
fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
    cu_seqlens=cu_seqlens)
```

## API Reference

### `flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, ...)`

| Parameter | Shape | Dtype | Description |
|---|---|---|---|
| `q` | `[B, T, H, D]` | bf16 | Query |
| `k` | `[B, T, H, D]` | bf16 | Key |
| `v` | `[B, T, H, D]` | bf16 | Value |
| `g` | `[B, T, H, D]` | bf16 | Gate (pre-activation) |
| `beta` | `[B, T, H]` | bf16 | Beta logits (pre-sigmoid) |
| `scale` | scalar | float | Scaling factor (typically `1/sqrt(D)`) |
| `out` | `[B, T, H, D]` | bf16 | Output buffer (written in-place) |
| `A_log` | `[H]` | fp32 | Log-gate parameter |
| `dt_bias` | `[H, D]` | fp32 | Gate bias |
| `lower_bound` | scalar | float | Gate lower bound (typically `[-5, 0]`) |
| `initial_state` | `[N, H, D, D]` | bf16/fp32 | Optional initial state |
| `final_state` | `[N, H, D, D]` | bf16/fp32 | Optional output state buffer |
| `cu_seqlens` | `[N+1]` | int64 | Optional cumulative sequence lengths |

**Constraints**: D must be 128. All inputs must be NPU, contiguous. B=1 when cu_seqlens provided.

## FLA Integration

FlashKDA auto-dispatches from `fla.ops.kda.chunk_kda` when installed:

```python
import flash_kda.fla_integration  # register dispatch
from fla.ops.kda import chunk_kda  # now uses flash_kda on NPU
```

Opt out with `FLA_FLASH_KDA=0`. Debug with `logging.basicConfig(level=logging.INFO)`.

## Testing

```bash
# Quick smoke test
bash tests/test.sh

# Full parametrized test suite
bash tests/run_test_full.sh

# Single test
cd tests && pytest test_fwd_full.py -x -v -k "test_fwd_fixed"
```

## Architecture

Two-kernel pipeline, template-specialized on 5 boolean flags (`HasStateIn`, `HasStateOut`, `StateFP32`, `IsVarlen`), producing 14 variants.

### Kernel 1: Prepare (AIV-dominant)

Grid: `(total_tiles × H)`, 1 AIC+AIV core per (tile, head)

1. **AIV**: Load q, k, g, beta, dt_bias from GM → UB
2. **AIV**: L2 normalize q and k (vector reduction)
3. **AIV**: Gate activation + cumulative sum
4. **AIV**: Decay: k_decayed, q_decayed, k_inv, k_restored
5. **AIV**: Store workspace to GM
6. **AIC**: Compute L = k_decayed @ k_inv^T, Mqk = q_decayed @ k_inv^T (MMAD)
7. **AIV**: Apply lower-triangular mask + beta sigmoid, construct (I - L)
8. **AIC**: Neumann inverse INV = (I-L)^{-1} (6 MMADs)
9. **AIV**: Store INV and Mqk to workspace

### Kernel 2: Recurrence (AIC-dominant)

Grid: `(N × H)`, 1 AIC+AIV core per (sequence, head), 2-stage pingpong pipeline

1. **AIC**: Load initial state from GM → L1
2. Per-chunk loop:
   - **AIC**: Dual GEMM k_decayed@state, q_decayed@state
   - **AIV**: v_sub = (v - k_decayed@state) * sigmoid(beta)
   - **AIC**: u = INV @ v_sub, out = q_decayed@state + Mqk @ u
   - **AIC**: State update GEMM k_restored^T @ u
   - **AIV**: state = state * exp(g_total) + k_restored^T @ u
   - **AIV**: Store output to GM
3. **AIC**: Store final state to GM

### Memory Hierarchy

```
GM ──Nd2Nz──→ L1 (512KB) ──LoadData──→ L0A/L0B (64KB) ──MMAD──→ L0C (128KB) ──Fixpipe──→ UB/GM
     MTE2_MTE1           M_MTE1                          M_FIX            FIX_M
```

### Key Differences from CUDA Version

| CUDA (SM90) | Ascend (Atlas A2) |
|---|---|
| TMA (Tensor Memory Accelerator) | DataCopy GM→L1 (Nd2Nz format conversion) |
| Warp-level MMA (SM80_16x8x16) | TileMmad (16x16x16 per AIC core) |
| PipelineTmaAsync | HardEvent flags + 2-stage pingpong |
| PipeBarrier (warp sync) | CrossCoreSetFlag/WaitFlag (AIC↔AIV) |
| SM75_U32x1_MOVM_T (register transpose) | LoadData2DParams.ifTranspose |
| GMMA swizzled workspace | RowMajor workspace + Nd2Nz on load |
| 4 MMA warps (192 threads) | 1 AIC core + 1 AIV core |

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN SDK path | Required |
| `FLASH_KDA_ASCEND_ARCH` | Target architecture | `2201` |
| `FLA_FLASH_KDA` | Set `0` to opt out of FLA dispatch | `1` |

## License

MIT
