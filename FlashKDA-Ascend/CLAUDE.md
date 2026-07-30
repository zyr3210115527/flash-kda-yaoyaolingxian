# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashKDA-Ascend: Flash Kimi Delta Attention on Huawei Ascend NPU. Reimplementation of [FlashKDA](https://github.com/MoonshotAI/FlashKDA) using catlass (CANN Templates for Linear Algebra Subroutines). Targets Atlas A2/A3 (CATLASS_ARCH=2201) and Ascend 950 (CATLASS_ARCH=3510).

## Build & Install

```bash
git submodule update --init --recursive
pip install -v --no-build-isolation .
FLASH_KDA_ASCEND_ARCH=3510 pip install -v --no-build-isolation .  # Ascend 950
```

Requires `ASCEND_HOME_PATH` env var pointing to CANN SDK.

## Tests

```bash
bash tests/test.sh
python tests/test_fwd.py                    # correctness vs torch_ref
```

## Architecture

Two-kernel pipeline. NOT a port — Ascend hardware model is fundamentally different from NVIDIA.

### Kernel 1: Prepare (AIV-dominant)
- Grid: (total_tiles, H)
- AIV: L2 normalize, gate activation, cumulative sum, decay, tril mask
- AIC: 3× 16×16 MMA (L, Mqk, Neumann inverse)
- AIC-AIV sync: CrossCoreSetFlag/WaitFlag (NOT PipeBarrier)

### Kernel 2: Recurrence (AIC-dominant)
- Grid: (N, H)
- AIC: Pingpong MMAD pipeline (dual GEMM, INV@u, k_restored^T@u)
- AIV: Element-wise (beta sigmoid, state * exp(g_total))
- Pipeline: 2-stage L1 double-buffer with HardEvent flags

### Key Mappings (CUDA → Ascend)

| CUDA Concept | Ascend Equivalent |
|---|---|
| TMA load/store | DataCopy GM↔L1/UB + HardEvent sync |
| PipelineTmaAsync | BlockMmadPingpong + SetFlag/WaitFlag |
| SM80 warp MMA | TileMmad (L0A/L0B→L0C) |
| SMEM | L1 (512KB) + UB (192KB) |
| Registers | L0A/L0B (64KB each) + L0C (128KB) |
| Warp shuffle | AIV vector ops (natural parallelism) |
| LDSM/STSM | CopyL1ToL0A/L0B, CopyL0CToUB |
| GMMA swizzle | Fractal layouts (zN, nZ) |
| __syncthreads | PipeBarrier / CrossCoreSetFlag |
| cudaStream_t | aclrtStream |

### Source Layout

- `flash_kda/__init__.py` — Python API wrapping C extension
- `src/flash_kda.cpp` — pybind11 binding, input validation, ACL runtime launch
- `include/flash_kda/layout.hpp` — Constants, params struct, workspace layout
- `include/flash_kda/utils.hpp` — Math approximations, workspace size computation
- `include/flash_kda/fwd.h` — Kernel launch declarations
- `include/flash_kda/fwd_kernel1.hpp` — Kernel 1 (prepare)
- `include/flash_kda/fwd_kernel2.hpp` — Kernel 2 (recurrence)
- `catlass/` — Git submodule

### Memory Hierarchy

```
GM (Global) → L1 (512KB, per-core) → L0A/L0B (64KB each, Cube input) → L0C (128KB, Cube accumulator) → UB (192KB, Vector buffer) → GM
```

Data flow for MMAD: GM→L1 (DataCopy, MTE1_MTE2) → L0A/L0B (DataCopy, M_MTE1) → MMAD → L0C → UB (CopyL0CToUB, M_FIX) → GM (DataCopy, FIX_M)

## Environment Variables

| Variable | Purpose |
|---|---|
| `ASCEND_HOME_PATH` | CANN SDK location (required) |
| `FLASH_KDA_ASCEND_ARCH` | `2201` (A2/A3, default) or `3510` (950) |

## Constraints

- D must be 128 (K = V = 128)
- All inputs: NPU, contiguous, bf16 (except A_log/dt_bias fp32, cu_seqlens int64)
- State tensors: bf16 or fp32; both present must match dtype
