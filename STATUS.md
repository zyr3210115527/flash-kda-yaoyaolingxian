# Status

Last updated: 2026-07-30

## Summary

The inherited `FlashKDA-Ascend` port **has never been compiled**. It is a detailed
sketch of the right algorithm written against an imagined version of the AscendC
and catlass APIs. Two independent audits — one per kernel, both cross-checked
against the vendored catlass sources and the CUDA reference — agree that the two
kernel files need to be rewritten rather than repaired.

This document records what is actually true about the code, so nobody has to
rediscover it.

## What the inherited tree claimed vs. what is there

`FlashKDA-Ascend/README.md` and `CLAUDE.md` describe a finished, tuned
implementation: "Full Implementation", a two-stage pingpong pipeline,
"template-specialized on 5 boolean flags producing 14 variants", and a table of
CUDA→Ascend mappings. None of that is backed by the code.

| Claim | Reality |
|---|---|
| Kernels are implemented | ~2000 lines exist; no `__global__` entry point, so nothing can be launched |
| `src/flash_kda.cpp` does "ACL runtime launch" | It calls `launch_fwd_prepare` / `launch_fwd_recurrence`, whose bodies are entirely comments |
| 14 template variants on 5 flags | Zero template specialization; the flags are runtime `int` fields. (The sentence also says "5 flags" and then lists 4.) |
| 2-stage L0C pingpong | L0C is always addressed at offset 0; the pingpong exists only in variable names |
| Builds with `pip install .` | CMake omits `find_package(ASC)`, so configuration fails before compiling anything |

There were no TODO or WIP markers anywhere to signal this.

## Blocking defects, by category

### The code cannot compile

- `layout.hpp` used `ArchTag::UBSize`, `L1Size`, `L0ASize`, `L0BSize`, `L0CSize`.
  The real names in `catlass/include/catlass/arch/arch.hpp` are `UB_SIZE`,
  `L1_SIZE`, `L0A_SIZE`, `L0B_SIZE`, `L0C_SIZE`. **Fixed.**
- `layout.hpp` defined `using BF16 = ascendFloat16` with a comment claiming
  AscendC uses `half` for bf16 on A2. `ascendFloat16` does not exist anywhere in
  CANN or catlass, and the claim is false — bf16 is `bfloat16_t` and Atlas A2
  supports it natively, including the Fixpipe `F322BF16` quantization mode.
  Both kernels then `reinterpret_cast` bf16 pointers to `half*` throughout, which
  reinterprets an 8-bit exponent as a 5-bit one: garbage, not rounding error.
  **Fixed in `layout.hpp`; the kernels still need it.**
- `utils.hpp` functions were plain `inline` but called from `__aicore__` code.
  **Fixed.**
- `TBuf<PIPE_V>` / `TBuf<PIPE_MTE1>` — `TBuf`'s parameter is an
  `AscendC::TPosition`, not a `pipe_t`.
- `TBuf::GetBufferAddr<T>(offset)` is called ~40 times and does not exist.
  The catlass equivalent is `Arch::Resource<ArchTag>` plus
  `LocalTensorBuffer::GetBufferByByte<T>(offset)`.
- `pipe.InitBuffer(buf, PIPE_X, size)` — the `TBuf` overload takes two arguments.
- `LocalTensor` has no `operator+`; `v_ub + row * D` must be `v_ub[row * D]`.
- Namespace-scope explicit specializations of `operator()` without `inline`
  would multiply-define as soon as two translation units include the header.

### The code assumes hardware that does not exist on Atlas A2

This is the part that makes it a rewrite rather than a repair.

- **AIC has no access to UB.** Both kernels have the AIC read and write `ubBuf`,
  `Fixpipe` into UB tensors, `DataCopy` UB→L1, and even run vector `Add`
  instructions on the AIC. On A2 the vector unit and UB belong to the AIV.
  `catlass/include/catlass/gemm/tile/copy_l0c_to_ub.hpp` is wrapped entirely in
  `#if CATLASS_ARCH == 3510` — the L0C→UB path is Ascend 950 only.
  Every AIC↔AIV hand-off must go through GM, as it does in
  `catlass/examples/19_mla` and `23_flash_attention_infer`.
- **There is no L1↔UB path at all**, in either direction.
- **AIV block index is a sub-core index.** One AIC pairs with two AIVs, so the
  AIV must use `GetBlockIdx() / GetSubBlockNum()` to agree with its AIC. As
  written, the two AIVs compute different tiles than their AIC, and the
  CrossCore flag counts come out 2:1 — wrong results and a hang.
- **`SetSyncBaseAddr(hardwareSyncAddr)` is missing.** It must be the first
  statement of the kernel entry, and the address comes from
  `aclrtGetHardwareSyncAddr` on the host. Without it every CrossCore
  set/wait is inert.
- **Guaranteed deadlock in kernel 1:** the AIV returns early for out-of-range
  varlen tiles before signalling, while its AIC is already blocked waiting.
- **Guaranteed deadlock in kernel 2:** `M_FIX` is set once at entry but waited
  on ~96 times per chunk; `MTE1_MTE2` is waited on every chunk but never set
  inside the loop.

### The numerics are wrong even where the code would run

- **Decay exponents are inflated by log2(e) ≈ 1.4427.** The CUDA kernel uses
  `ex2` (2^x) and folds `log2(e)` into `gate_scale` on the host. The port kept
  the folded `gate_scale` but switched to the natural `Exp`. **Fixed** by passing
  the raw `lower_bound` and using `Exp` consistently.
- **`inv_cumsum` is identically 1.** It aliases `exp_cumsum`, overwrites it with
  `Duplicate(..., 1.0f)`, then divides that by itself. Decay is silently lost.
- **`Exp(g_total)` is applied twice** — kernel 1 already exponentiates it before
  storing, matching the CUDA reference, and kernel 2 exponentiates again.
- **Phase A and Phase B UB offsets are identical**, so `decay_apply_aiv` reads
  `q_ub[row]` and writes `k_decayed_ub[row]` at the same address. The CUDA
  version can union these buffers only because it stages everything through
  registers first.
- **`Nd2NzParams` strides are swapped** (`dstNzC0Stride` and `dstNzNStride`).
- **The zN fractal offset helper is transposed.** It computes a zZ (row-major
  between fractals) offset. Harmless for the 16-row matrices, silently wrong for
  the [128,128] state. **Fixed in `utils.hpp`.**
- **`ifTranspose = true` on a zN source has no corresponding catlass path.**
  Transposing `k_restored` must be done at the GM→L1 step by loading the same
  bytes as ColumnMajor→nZ.
- **16×16 sub-blocks are flat-copied out of a [16,128] RowMajor UB buffer**,
  which picks up the wrong 256 elements — rows are 128 apart, not contiguous.
- The recurrent state is stored to GM transposed relative to the reference
  implementation's `[value, key]` convention.

## What was salvageable

Not everything is wrong, and the algorithm intent is a useful specification:

- The lower-triangular mask, beta sigmoid, and `(I - L)` construction in
  kernel 1 match the CUDA reference line for line, including the diagonal
  handling.
- The Neumann series decomposition `INV = (I-L)(I+L²)(I+L⁴)(I+L⁸)` is correct.
- The varlen tile-index derivation matches the reference.
- The runtime-flag design (rather than 14 template variants) is a reasonable
  choice — the README is what needs correcting.
- Accumulating K in fp32 inside L0C and adding the two output terms in bf16
  both match the reference's precision choices.

## Current state of this repository

Fixed so far:

- `include/flash_kda/layout.hpp` — rewritten: correct arch constants, `bfloat16_t`,
  GM_ADDR params, workspace extended with AIC↔AIV scratch slots.
- `include/flash_kda/utils.hpp` — rewritten: `__aicore__` device helpers, correct
  zN fractal addressing, correct Nd2Nz strides, workspace size derived from
  `WorkspaceSizes` so host and device cannot drift.
- `CMakeLists.txt` — rewritten against catlass's own torch-extension template:
  `find_package(ASC)` before `project()`, catlass as an include directory rather
  than `add_subdirectory`, `--npu-arch=dav-<arch>`, and linking against torch,
  torch_npu, `ascendc_runtime`, `runtime`, `ascendcl`, `tiling_api`, `platform`.

Still to do:

- `include/flash_kda/fwd_kernel1.hpp` — rewrite.
- `include/flash_kda/fwd_kernel2.hpp` — rewrite.
- `src/fwd_kernel1.asc` / `src/fwd_kernel2.asc` — kernel entry points and launch.
- `src/flash_kda.cpp` — pass the hardware sync address; drop the `is_cuda` naming.
- `README.md` / `CLAUDE.md` — correct the claims listed at the top of this file.

## Hardware

Development and debugging run on Cybertron's Wulanchabu cluster, which has
Ascend 910B (Atlas A2, `CATLASS_ARCH=2201`) — the exact target this port was
written for. HiDevLab is not needed.

Devspace job 645140: 1× 910B3, preemptable, CANN 8.5.0, `bisheng` and `ccec`
present, `ASCConfig.cmake` available so `find_package(ASC)` resolves.

The working rule from here: **nothing is described as done until it has compiled
and run on that card.** The state this project was inherited in is what happens
otherwise.
