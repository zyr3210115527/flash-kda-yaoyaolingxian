# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashKDA (Flash Kimi Delta Attention) — high-performance CUDA kernel for KDA attention, built on CUTLASS. Targets SM90+ (Hopper). Developed by MoonshotAI. Also serves as a backend for `flash-linear-attention` (FLA), auto-dispatched from `fla.ops.kda.chunk_kda`.

## Build & Install

```bash
git submodule update --init --recursive   # fetches CUTLASS
pip install -v --no-build-isolation .     # auto-detects current GPU arch
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .  # all archs (CI/wheels)
```

Supported archs: `90a`, `100a`, `103a`, `120a`. Env var `FLASH_KDA_CUDA_ARCHS` accepts `auto` (default), `all`, or comma-separated list.

## Tests

```bash
bash tests/test.sh                          # quick: install + test_fwd.py
bash tests/run_test_full.sh                 # full: pytest -n 16 with GPU distribution
cd tests && pytest test_fwd_full.py -x -v -k "test_fwd_fixed"   # single test group
cd tests && pytest test_fwd_full.py::test_fwd_fixed[T16-H1-bf16-in+out]  # single parametrized case
python tests/test_fwd.py                    # 4 top-level functions (no pytest needed)
```

Set `FLASH_KDA_DIST_GPU=1` for multi-GPU pytest-xdist runs.

## Architecture

Two-kernel pipeline, template-specialized on 5 boolean flags (`HasStateIn`, `HasStateOut`, `StateFP32`, `IsVarlen`), producing 14 variants.

### Data Flow

1. **Kernel 1** (`_flash_kda_fwd_prepare`, `csrc/smxx/fwd_kernel1.cuh`) — Grid `(total_tiles, H)`, 256 threads
   - TMA loads q, k, g, beta, dt_bias
   - L2 normalize q and k
   - Fused gate activation: `g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`
   - Cumulative sum of g; compute `k_decayed`, `q_decayed`, `k_inv`, `k_restored`
   - Build `L = k_decayed @ k_inv^T`, `Mqk = q_decayed @ k_inv^T`
   - Apply beta sigmoid to L (lower triangular)
   - Neumann series inverse: `INV = (I - L)^{-1}` in fp16
   - TMA store intermediates to workspace

2. **Kernel 2** (`_flash_kda_fwd_recurrence`, `csrc/smxx/fwd_kernel2.cuh`) — Grid `(N, H)`, 192 threads (4 MMA + 1 load + 1 store warp)
   - Load initial state; pipelined loop over chunks
   - Dual GEMM: `k_decayed @ state`, `q_decayed @ state`
   - `u = INV @ ((v - k_decayed @ state) * beta)`
   - `out = q_decayed @ state + Mqk @ u`
   - State update: `state = state * exp(g_total) + k_restored^T @ u`
   - TMA store output and final state

### Source Layout

- `flash_kda/__init__.py` — Python API: `flash_kda.fwd()` wraps C++ extension, allocates workspace
- `csrc/flash_kda.cpp` — pybind11 binding, input validation, dispatch to `launch_fwd<>` template
- `csrc/fwd.h` — Template declaration for `launch_fwd<>`
- `csrc/smxx/fwd_launch.cu` — TMA setup, kernel launch, explicit template instantiation
- `csrc/smxx/fwd_kernel1.cuh` — Kernel 1 (prepare)
- `csrc/smxx/fwd_kernel2.cuh` — Kernel 2 (recurrence)
- `csrc/smxx/utils.cuh` — Shared: layouts, pipelines, MMA helpers, Neumann inverse, fp32/bf16 conversion
- `cutlass/` — Git submodule (NVIDIA CUTLASS)
- `tests/torch_ref.py` — Reference implementation matching kernel numerics exactly (same sigmoid, fp16 accumulation, L2 norm)

### Workspace

Single contiguous buffer allocated in Python, partitioned into 6 arrays: `k_decayed`, `q_decayed`, `k_restored`, `g_total`, `INV`, `Mqk`. Size computed by `get_workspace_size(T_total, H, N)`.

## Key Constraints

- D must be 128 (K = V = 128)
- All inputs: CUDA, contiguous, bf16 (except A_log/dt_bias fp32, cu_seqlens int64)
- State tensors: bf16 or fp32; both present must match dtype
- Varlen: B must be 1, cu_seqlens shape `[N+1]`, state shape `[N, H, D, D]`
- Batched: state shape `[B, H, D, D]`

## Environment Variables

| Variable | Purpose |
|---|---|
| `FLASH_KDA_CUDA_ARCHS` | `auto`/`all`/comma-separated arch list |
| `NVCC_THREADS` | nvcc thread count (default 32) |
| `FLASH_KDA_VERSION_SUFFIX` | Override version suffix |
| `FLA_FLASH_KDA` | Set `0` to opt out of FLA auto-dispatch |
| `FLASH_KDA_DIST_GPU` | Set `1` for pytest-xdist GPU assignment |

## Development Setup

```bash
bash setup_clangd.sh   # generates .clangd for IntelliSense on CUDA sources
```

## FLA Integration

FlashKDA auto-dispatches from `fla.ops.kda.chunk_kda` when installed. Must call under `torch.inference_mode()`. Opt out with `FLA_FLASH_KDA=0`. Debug dispatch with `logging.basicConfig(level=logging.INFO)`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **FlashKDA** (185 symbols, 303 relationships, 6 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "master"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/FlashKDA/context` | Codebase overview, check index freshness |
| `gitnexus://repo/FlashKDA/clusters` | All functional areas |
| `gitnexus://repo/FlashKDA/processes` | All execution flows |
| `gitnexus://repo/FlashKDA/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
