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
python tests/test_all_parse.py     # every file parses. No hardware, <1s. Run first.
python tests/test_shapes.py        # 12 shape and feature cases vs the CPU oracle
python tests/race_probe.py         # determinism. Run it 6+ times; once proves nothing.
python benchmarks/ab_compare.py    # interleaved A/B, resolves ~1%
python benchmarks/profile_pipes.py # hardware pipe counters
```

Never conclude a change is neutral from one timed run: between-process spread is
±8%, within-process is 0.4–0.9%. `ab_compare.py` exists for that reason.

## Architecture

Two-kernel pipeline. NOT a port — the Ascend hardware model is fundamentally
different from NVIDIA. Both kernels are **single fused launches** using both
vector cores; each phase declares its core type with
`if constexpr (g_coreType != ...)`.

### Kernel 1: Prepare (AIV-dominant), grid-strided over (tile, head)
- AIV: L2 normalize, gate activation, cumulative sum, decay, tril mask
- AIC: 3× 16×16 MMA (L, Mqk, Neumann inverse — the chain stays resident in L1)
- Both AIV subblocks participate, splitting each phase's rows

### Kernel 2: Recurrence (AIC-dominant), one unit per (sequence, head)
- AIC: `PostGemms(c)` then `PreGemms(c+1)` in one turn — 2 handshakes/chunk
- AIV: the `[D,D]` recurrent state stays resident in UB across the whole loop
- `StoreOut` ∥ `DecayState` on an uneven 8/20 row split, tuned by measurement

### Non-negotiable rules, each of which cost days

- **Every kernel entry needs `KERNEL_TASK_TYPE_DEFAULT(...)`.** Without it the
  runtime never provisions the FFTS sync resources and the first
  `CrossCoreWaitFlag` hangs forever, with no diagnostic. `FlashKdaSyncOneArg`
  must stay `MIX_AIC_1_1` — it has a single AIV path and deadlocks under 1_2.
- **`CrossCoreSetFlag<0x2>` is per-AI-Core.** Arming is per flagId but
  *collective* over the core's AIV group: a wait on X is satisfied only once
  every AIV has set X. So work assignment must keep each core's AIVs on their
  own AIC's units, and no subblock can arm a flag alone.
- **`M_MTE1` after every `Mmad`**, or the device intermittently reports
  "L0B read/write conflict in the MTE".
- **`FIX_MTE2` before every `Gemm`'s loads.** It looks removable in `PreGemms`
  and is not — see docs/debugging-notes.md, it costs correctness for 3%.
- The AIC cannot reach UB on A2. AIC↔AIV data goes through GM.

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

- `flash_kda/__init__.py` — Python API wrapping the C extension; **caches the
  workspace**. Reallocating it per call was once 88–98% of measured time.
- `src/flash_kda.cpp` — pybind11 binding, input validation, ACL runtime launch.
  Also exports the workspace layout (`CHUNK`, `WS_PER_TILE`, `WS_SLOT_OFF`, …)
  so probes never hardcode it, and the sync probes (`sync_onearg`, `persub_flag`).
- `src/fwd_kernel1.asc`, `src/fwd_kernel2.asc` — the kernel **entry points**.
  This is where `KERNEL_TASK_TYPE_DEFAULT` is declared; edit these annotations
  one at a time, never with a global `sed` (that flipped `FlashKdaSyncOneArg`
  to 1_2 once and deadlocked it).
- `include/flash_kda/layout.hpp` — constants, params struct, workspace layout.
  `SlotOffset`/`NarrowIndex` map slot index → byte offset; slots are two wide
  `[D,D]` plus seven narrow `[CHUNK,D]`, ordered 0,1,2,4,6,5,8 so 4 and 6 are
  adjacent.
- `include/flash_kda/utils.hpp` — math approximations, workspace size
- `include/flash_kda/fwd.h` — kernel launch declarations
- `include/flash_kda/fwd_kernel1.hpp` — kernel 1 (prepare)
- `include/flash_kda/fwd_kernel2.hpp` — kernel 2 (recurrence)
- `tests/torch_ref.py` — the CPU oracle. Bit-exact against the CUDA kernel's
  arithmetic, and itself verified against independent float64 by
  `tests/validate_ref.py`. Everything else is checked against this.
- `catlass/` — git submodule

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
| `FLASH_KDA_FUSED_K1`, `FLASH_KDA_FUSED_K2` | `=0` falls back to per-phase launches |
| `FLASH_KDA_L1_NEUMANN` | `=0` routes the Neumann chain through GM |
| `FLASH_KDA_FAST_NORM` | `=0` reduces a row at a time |
| `FLASH_KDA_SKIP_K2` | kernel1 only. **Required** when a probe reads kernel1's workspace slots, or kernel2 will have already reused them. |

Each optimisation keeps its predecessor reachable, because that predecessor is
what it was validated against. All four fallbacks must stay at 12/12.

## Constraints

- D must be 128 (K = V = 128)
- All inputs: NPU, contiguous, bf16 (except A_log/dt_bias fp32, cu_seqlens int64)
- State tensors: bf16 or fp32; both present must match dtype
- `CHUNK · |lower_bound| ≤ ~88` — the in-chunk decay must stay in fp32 range.
  A model constraint the CUDA reference shares, not a kernel choice.

## Working notes

- Probes must read the workspace layout from the extension (`_C.CHUNK`,
  `_C.WS_SLOT_OFF`, `_C.WS_PER_TILE`), never hardcode offsets. They have moved
  twice, and a stale copy reads the wrong tile while looking plausible.
- `struct` has no bf16 format code and `'e'` is IEEE fp16 — reading bf16
  workspace fields with `'e'` makes correct data look wrong by 20–500×.
- `DataCopyPad` pads every block to 32 bytes, so gathering one bf16 per token
  does not land them contiguously.
- The full evidence trail for every non-obvious behaviour is in
  `../docs/debugging-notes.md`, including the hypotheses that were eliminated.
