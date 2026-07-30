# flash-kda-yaoyaolingxian

Porting **FlashKDA** (Flash Kimi Delta Attention) from NVIDIA CUTLASS to Huawei Ascend NPU, using **catlass** (CANN Templates for Linear Algebra Subroutines — Huawei's CUTLASS counterpart) and Ascend C.

## Layout

| Path | What it is |
|---|---|
| `FlashKDA/` | Upstream CUDA reference implementation, vendored from [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) at `a8c7335` (2026-07-28) |
| `FlashKDA-Ascend/` | The Ascend C port — the actual work |
| `FlashKDA-Ascend/catlass/` | Vendored from [atomgit.com/cann/catlass](https://atomgit.com/cann/catlass) at `b71539e` (2026-07-28) |
| `STATUS.md` | Honest assessment of what works and what doesn't |
| `docs/hidevlab.md` | Bring-up notes for building and debugging on HiDevLab free NPU cards |

Both upstreams are vendored as plain files rather than submodules so the tree is self-contained — a build host that cannot reach github.com or atomgit.com still gets everything it needs.

## Status

**The port does not build or run yet.** It was inherited as an unfinished draft: roughly 2000 lines of Ascend C kernel logic exist, but there is no kernel entry point, the launch functions are empty, and the build system does not link the Ascend runtime. See `STATUS.md` for the itemized gap list.

Nothing in this repo has been executed on real hardware yet. Treat every performance or correctness claim in the inherited `FlashKDA-Ascend/README.md` and `CLAUDE.md` as aspirational until a test run on an Atlas A2/A3 card says otherwise.

## Target hardware

- Atlas A2 / A3 (`CATLASS_ARCH=2201`)
- Ascend 950 (`CATLASS_ARCH=3510`)
- CANN SDK >= 8.5.0, PyTorch + torch_npu

## The algorithm

KDA forward is a chunked linear-attention recurrence. Per chunk of 16 tokens:

```
out   = q_decayed @ state + Mqk @ (INV @ ((v - k_decayed @ state) * sigmoid(beta)))
state = state * exp(g_total) + k_restored^T @ u
```

where `INV = (I - L)^{-1}` is computed by Neumann series, and `L`, `Mqk`, `k_decayed`, `q_decayed`,
`k_restored`, `g_total` are per-chunk intermediates produced by a prepare pass. The port splits
these into two kernels: prepare (vector-dominant, AIV) and recurrence (cube-dominant, AIC).
