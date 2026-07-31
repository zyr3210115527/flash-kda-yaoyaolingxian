# Forward benchmark — Ascend 910B

Measured on one Ascend 910B3, CANN 8.5.0, bf16, D=128. Wall clock around a
synchronize (torch_npu's event API needs operator kernels this image lacks),
3 warmup + 10 iterations, **workspace allocated once outside the timing loop**
(`benchmarks/bench_hoisted.py`).

That last point is the whole story of the first round of measurements. The
original harness called the `flash_kda.fwd` wrapper, which allocates the
workspace on every call:

    workspace = torch.zeros(get_workspace_size(...), dtype=uint8).to(device)

At T=1024 H=8 that is 302 MB of host-side zeros copied to the device per call.
It accounted for 88–98% of every number in the first version of this file. The
tell was that stubbing out every phase of kernel1 changed the total by 2%.

## Measurements

| T | H | workspace | end-to-end ms | kernel1 ms | µs/token/head |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 154 M | 1.40 | 0.40 | 0.3410 |
| 1024 | 8 | 302 M | 2.73 | 0.76 | 0.3330 |
| 2048 | 8 | 599 M | 5.41 | 1.47 | 0.3292 |
| 2048 | 16 | 1198 M | 7.55 | 2.90 | 0.2307 |
| 4096 | 8 | 1193 M | 10.83 | 2.90 | 0.3309 |

End-to-end at T=1024 H=8 started this round at 29.96 ms and is now 2.73 ms.

## Against the CUDA version

At the CUDA implementation's own benchmark shapes, which are now reachable --
they were not before, because the wrapper built the workspace as host-side
zeros and copied it, and ~20 GB does not go through the host:

| T | H | workspace | ours ms | CUDA/H20 ms | ratio |
|---:|---:|---:|---:|---:|---:|
| 8192 | 64 | 18.6 G | 94.53 | 1.6217 | **58×** |
| 8192 | 96 | 27.9 G | 136.54 | 2.6220 | **52×** |

Same shape, same dtype, wall clock both sides. Our µs/token/head flattens at
about 0.180 from H=64 onward, so this is a stable figure rather than one that
keeps narrowing with size.

For reference, the same number earlier in this work was 790× — that came from
a harness where per-call workspace allocation was 88–98% of the measurement.
Correcting the methodology brought it to ~94×; the optimizations below and
reaching the matching shape brought it to 58×.

The remaining caveat is the one that cannot be measured away here: different
silicon. An H20 and a 910B3 are not the same machine, so this is a scale
reference, not a statement about the two implementations' quality. The
comparison that would isolate the code from the hardware is a PyTorch baseline
on this same card, which is what the next section is about.

## A PyTorch baseline would be the honest comparison, and cannot run here

Comparing against CUDA-on-H20 conflates the code with the hardware. The
question worth answering is whether this kernel beats a straightforward
PyTorch implementation *on the same card*. `benchmarks/bench_torch_baseline.py`
implements that, but it cannot run on this devspace: every aclnn operator
fails with error 561103, including `add` and `matmul`. The image installs the
CANN toolkit without the binary operator package —

    $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend910b/

contains only `ops_oam`. Our kernels are unaffected because `.asc` files are
compiled directly and never dispatch through that path, which is also why the
correctness suite works (its reference runs on CPU). Running the baseline
needs an image with `Ascend-cann-kernels-910b`.

## Where the time goes

Kernel1, phase by phase, cumulative (each phase stubbed out and added back):

| | ms |
|---|---:|
| launch only, all phases stubbed | 0.06 |
| + Prepare (AIV) | 0.28 |
| + LMqk (AIC, 2 gemms) | 0.30 |
| + Mask (AIV) | 0.47 |
| + Neumann (AIC, 16 gemms) | 0.98 |

Kernel2, same method (kernel1 held at a fixed 0.76 ms):

| | ms |
|---|---:|
| kernel1 + 322 empty kernel2 launches | 2.610 |
| + BuildU (AIV) | 2.498 |
| + PreGemms (AIC) | 2.684 |
| + FinishOut (AIV) | 2.692 |
| + PostGemms (AIC) | 2.711 |
| + FinishChunk (AIV) | 3.096 |

**Kernel2 is 79% launch overhead.** 322 launches at ~5.7 µs is 1.85 ms; all
five phases together cost 0.49 ms. The thing to cut there is launch count, not
work.

## What was tried

Measured, in order, with the outcome rather than the prediction:

1. **Batch the L2 normalization** — predicted the biggest win, delivered no
   measurable change. `NormalizeRow` ran 32 times per tile, each with a V_S/S_V
   round trip; batching to 2 flushes was correct but the flushes were not the
   bottleneck. Kept because it is the same arithmetic in less code.

2. **Grid-stride kernel1** — no change, reverted. blockDim was 512 against 20
   cube cores. Dispatching 512 blocks costs 0.55 ms with phases stubbed versus
   0.05 ms with 20, but total kernel1 time was 0.97 ms either way: dispatch
   overlaps with compute, so the stubbed figure is throughput in isolation, not
   an additive cost.

3. **Fuse the Neumann series** — worked: kernel1 0.97 → 0.76 ms, about 20%.
   `ComputeNeumann` issued 16 cube round trips and 10 were `X * I`, matrix
   copies done as 16×16×16 matmuls — a full GM→L1→L0→cube→L0C→GM trip with five
   barriers to move 512 bytes. Replaced with a fused `dst = A*B1 + A*B2` (two
   Mmads, one L0C accumulator, one Fixpipe) plus ping-pong buffers. 16 → 6.

4. **Fold BuildU into the previous chunk's FinishChunk** — worked: 3.18 → 2.81
   ms, about 12%. The per-chunk chain PreGemms(AIC) → FinishOut(AIV) →
   PostGemms(AIC) → FinishChunk(AIV) alternates core type at every arrow, so
   those boundaries are forced while cross-core sync is unavailable. BuildU of
   chunk t+1 is AIV and already ran right after FinishChunk of chunk t, so that
   boundary bought nothing. 5 launches per chunk became 4.

   Hoisting BuildU out of the loop entirely was tried first and broke every
   multi-chunk shape: it ends with `StateToBf16`, which reads the live
   recurrent state. The byte-identical error across three different fixes was
   the clue that the state read, not the v/beta work, pinned it to the loop.

5. **Stop reallocating the workspace** — the largest win of the round for real
   callers: 29.96 → 2.73 ms end-to-end, about 11×. The wrapper allocated and
   zeroed 302 MB on the host and copied it to the device on every call. The
   zeroing was a guard against reading a slot before writing it;
   `tests/workspace_poison.py` shows output is bit-identical with the workspace
   filled with NaN, so nothing does. Now allocated on device and cached.

## What to do next

1. **Move FinishOut to the cube.** It computes
   `u = (v − k_dec@state) · sigmoid(beta)`, a row-broadcast scale, which is
   `diag(beta) @ (v − k_dec@state)`. If it runs on AIC, PreGemms + FinishOut +
   PostGemms collapse into one launch and the chunk costs 2 launches instead of
   4 — roughly another 0.7 ms. The risk is precision, not structure: it forces
   `k_dec@state` through bf16 before the subtraction, and the delta rule makes
   `v − k_dec@state` a cancelling difference, so this needs measuring before
   it is believed.
2. **Raise CHUNK from 16.** Chunk count drives kernel2's launch count directly,
   and 16×16×16 is the cube's minimum fractal, so every gemm is one fractal of
   work behind full instruction overhead. CHUNK=64 would cut launches 4× and
   make each gemm 64 fractals. It is a large change: masks, buffer sizes, and
   two more Neumann factors.
3. **Vectorize `MaskAndBuild`** — builds a 16×16 mask with a scalar double
   loop, 256+ `GetValue`/`SetValue`. The Mask phase is 0.17 ms.
4. **Shrink the workspace.** `H × total_tiles × 608 KB`, because every scratch
   slot is sized for a `[128,128]` fp32 state when only kernel2's state slots
   need that. At T=8192 H=64 the layout wants ~20 GB, which is what blocks
   benchmarking at the CUDA version's shapes.

## A note on the numbers above

Every measurement here was taken after fixing a race in `Gemm128` (see
`tests/race_*.py`) under which only the last cube result before the kernel
ended was reliable. Before that fix the suite still passed, but the same shape
gave different errors on different runs, and `Mqk` was stably wrong. Timings
taken before it are not comparable.
