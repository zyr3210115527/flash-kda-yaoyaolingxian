# Forward benchmark — Ascend 910B

Measured on one Ascend 910B3, CANN 8.5.0, bf16, D=128. Wall clock around a
synchronize (torch_npu's event API needs operator kernels this image lacks),
3 warmup + 10 iterations.

**This is an untuned kernel.** Nothing has been optimized; correctness came
first. The numbers below are a starting point, not a result.

## Measurements

| T | H | chunks | k2 launches | workspace | ms | µs/token/head |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 8 | 42 | 21 M | 4.158 | 8.1220 |
| 256 | 4 | 16 | 82 | 40 M | 7.478 | 7.3023 |
| 512 | 4 | 32 | 162 | 77 M | 11.312 | 5.5233 |
| 512 | 8 | 32 | 162 | 154 M | 19.148 | 4.6748 |
| 1024 | 8 | 64 | 322 | 302 M | 27.250 | 3.3264 |
| 2048 | 8 | 128 | 642 | 599 M | 43.486 | 2.6542 |
| 2048 | 16 | 128 | 642 | 1198 M | 80.497 | 2.4566 |

## Against the CUDA version

The CUDA implementation's own benchmark, on an H20:

| Shape | ms | µs/token/head |
|---|---:|---:|
| T=8192 H=64 D=128 fixed | 1.6217 | 0.0031 |
| T=8192 H=96 D=128 fixed | 2.6220 | 0.0033 |

So roughly **790× slower per token-head** at our best measured shape.

Two caveats that make this a scale reference rather than a like-for-like
comparison: different hardware (910B vs H20), and different shapes — the
Ascend workspace does not currently reach T=8192 (see below), and our
µs/token/head is still falling with size, so the gap would narrow somewhat at
larger T.

## Where the time actually goes

Not where I expected. Breaking it down (`benchmarks/bench_breakdown.py`):

| T | H | kernel1 ms | full ms | kernel2 ms | launches | launch ms | launch % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 18.97 | 20.40 | 1.42 | 166 | 1.20 | 6% |
| 1024 | 8 | 28.83 | 30.16 | 1.33 | 326 | 2.35 | 8% |
| 2048 | 8 | 51.58 | 53.87 | 2.29 | 646 | 4.65 | 9% |

**kernel1 is ~93% of the runtime.** Kernel2 — the part with five launches per
chunk, the design I was most suspicious of — costs 1–2 ms. An empty kernel
launch measures 7.2 µs, so all the launches together are under 10%.

The cause is that kernel1's per-tile work is written as scalar and per-row
loops, each carrying a pipeline flush:

- `NormalizeRow` runs twice per row over 16 rows — 32 calls per tile, each a
  `Mul` + `ReduceSum` + `V_S` + scalar read + `S_V` + `Muls`. That is 32
  scalar/vector round trips per tile just to L2-normalize.
- the gate + cumsum loop is 16 sequential iterations with a `PipeBarrier` each
- `Decay` is 16 iterations of ~10 vector ops with 14 `PipeBarrier`s
- `MaskAndBuild` builds the 16×16 mask with a scalar double loop — 256+
  `GetValue`/`SetValue`
- `ComputeNeumann` issues 9 GEMMs, each a full Nd2Nz → LoadData → Mmad →
  Fixpipe round trip through GM

## What to do about it, in order

1. **Batch the L2 normalization.** All 16 rows at once instead of 32 separate
   reduce-and-scale round trips. Biggest single win available.
2. **Vectorize `MaskAndBuild`.** The tril mask and beta scaling are elementwise
   over a 16×16 tile; building them with scalar stores is the wrong shape
   entirely. `Duplicate` + a comparison mask would do it.
3. **Keep the Neumann iteration in L0C.** Nine GM round trips for nine 16×16
   MMADs is almost all overhead. `Mmad`'s accumulate mode can chain them.
4. **Fuse the decay loop.** It is 16 iterations over a `[16,128]` tile that
   could largely be single whole-tile vector ops.
5. **Shrink the workspace.** It is `H × total_tiles × 608 KB` because every
   scratch slot is sized for a `[128,128]` fp32 state, but only kernel2's state
   slots need that, and those are per (sequence, head) rather than per (tile,
   head). At T=8192 H=64 the current layout would want ~20 GB. Fixing this is
   what unblocks benchmarking at the CUDA version's shapes.
6. Only then worry about launch count.
