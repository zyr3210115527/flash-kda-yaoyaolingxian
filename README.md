# flash-kda-yaoyaolingxian

Porting **FlashKDA** (Flash Kimi Delta Attention) from NVIDIA CUTLASS to Huawei
Ascend NPU, using **catlass** (CANN Templates for Linear Algebra Subroutines —
Huawei's CUTLASS counterpart) and Ascend C.

## Status

**The forward pass is correct and runs on hardware.** 12/12 shape and feature
cases match the CPU oracle at the bf16 noise floor, bit-identical across
repeated runs, with no faults or hangs.

At the CUDA implementation's own benchmark shape (T=8192, H=64) it is **20×**
slower than that implementation on an H20 — 32.58 ms against 1.62 ms. It started
at 790×, though most of that first figure was a measurement artifact rather
than real slowness; see [BENCHMARK.md](BENCHMARK.md) for the accounting of what
each step was actually worth.

Different silicon on the two sides, so treat 20× as a scale reference rather
than a verdict on either implementation.

## Layout

| Path | What it is |
|---|---|
| `FlashKDA-Ascend/` | The Ascend C port — the actual work |
| `FlashKDA/` | Upstream CUDA reference, vendored from [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) at `a8c7335` (2026-07-28) |
| `FlashKDA-Ascend/catlass/` | Vendored from [atomgit.com/cann/catlass](https://atomgit.com/cann/catlass) at `b71539e` (2026-07-28) |
| [`RETROSPECTIVE.md`](RETROSPECTIVE.md) | The five-day story: what was tried, what failed, and what Ascend is like to work on |
| [`BENCHMARK.md`](BENCHMARK.md) | Measurements, and what each optimisation was worth — including the ones worth nothing |
| [`STATUS.md`](STATUS.md) | What works, what doesn't, what is unresolved |
| [`docs/bringup.md`](docs/bringup.md) | Getting a build running on an Ascend card |
| [`docs/debugging-notes.md`](docs/debugging-notes.md) | Every non-obvious hardware behaviour found, with the evidence |
| `KIMI_NOTES*.md`, `KIMI_REPLY*.md` | Correspondence with Kimi (Moonshot AI), who supplied official docs and caught real problems |

Both upstreams are vendored as plain files rather than submodules, so a build
host that cannot reach github.com or atomgit.com still has everything.

## Target hardware

- Atlas A2 / A3 (`CATLASS_ARCH=2201`) — what this was developed and measured on
- Ascend 950 (`CATLASS_ARCH=3510`) — builds, untested
- CANN SDK ≥ 8.5.0, PyTorch + torch_npu

## Build

```bash
git submodule update --init --recursive
pip install -v --no-build-isolation .            # needs ASCEND_HOME_PATH
FLASH_KDA_ASCEND_ARCH=3510 pip install -v --no-build-isolation .   # Ascend 950
```

## Test

```bash
python tests/test_shapes.py          # 12 shape and feature cases vs the CPU oracle
python tests/race_probe.py           # determinism — run it several times, once is not enough
python tests/workspace_poison.py     # nothing reads the workspace before writing it
python benchmarks/ab_compare.py      # interleaved A/B, resolves ~1%
python benchmarks/profile_pipes.py   # hardware pipe counters
```

Correctness runs against `tests/torch_ref.py`, a bit-exact CPU emulator of the
CUDA kernel's arithmetic — its rounding, its fp16 accumulation, its sigmoid
approximation.

## Feature flags

Each optimisation keeps its predecessor reachable, because that predecessor is
what it was validated against. All four fallbacks pass 12/12.

| Variable | `=0` selects |
|---|---|
| `FLASH_KDA_FUSED_K2` | kernel2 as one launch per phase, instead of a single fused kernel |
| `FLASH_KDA_FUSED_K1` | kernel1 likewise |
| `FLASH_KDA_L1_NEUMANN` | the Neumann series through GM instead of staying in L1 |
| `FLASH_KDA_FAST_NORM` | row-at-a-time reduction instead of one whole-tile instruction |
| `FLASH_KDA_SKIP_K2` | kernel1 only, for profiling |

## The algorithm

KDA forward is a chunked linear-attention recurrence. Per chunk of 16 tokens:

```
out   = q_decayed @ state + Mqk @ (INV @ ((v - k_decayed @ state) * sigmoid(beta)))
state = state * exp(g_total) + k_restored^T @ u
```

`INV = (I - L)^{-1}` comes from a Neumann series, and `L`, `Mqk`, `k_decayed`,
`q_decayed`, `k_restored`, `g_total` are per-chunk intermediates from a prepare
pass. The port is two kernels: prepare (vector-dominant) and recurrence
(cube-dominant), each now a single fused launch that hands off between the cube
and vector cores with cross-core flags.

## Constraints

- `D` must be 128.
- Inputs on NPU, contiguous, bf16 (`A_log`/`dt_bias` fp32, `cu_seqlens` int64).
- State tensors bf16 or fp32; if both are present they must match.
- `CHUNK · |lower_bound| ≤ ~88` — the in-chunk decay has to stay in fp32 range.
  At `CHUNK=16` with `lower_bound=−5` that is 80: inside, but not by much. The
  CUDA reference shares this bound and is also at `CHUNK=16`.
