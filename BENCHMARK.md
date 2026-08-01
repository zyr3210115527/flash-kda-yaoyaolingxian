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
| 8192 | 64 | 18.6 G | 80.03 | 1.6217 | **49×** |
| 8192 | 96 | 27.9 G | 113.72 | 2.6220 | **43×** |

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

1. **Batch the L2 normalization** — no measurable change. Kept (same
   arithmetic, less code), but chosen by reasoning rather than measurement,
   which is why it missed.

2. **Grid-stride kernel1, first attempt** — no change, reverted. Correct at the
   time: compute still dominated, so cutting dispatch changed nothing.

3. **Fuse the Neumann series** — kernel1 0.97 → 0.76 ms. 16 cube round trips,
   10 of them `X * I` matrix copies done as 16×16×16 matmuls, became 6 via a
   fused `dst = A*B1 + A*B2`.

4. **Fold BuildU into the previous chunk's FinishChunk** — 3.18 → 2.81 ms at
   T=1024 H=8. 5 launches per chunk became 4.

5. **Stop reallocating the workspace** — 29.96 → 2.73 ms end-to-end, ~11×. The
   wrapper zeroed 302 MB on the host and copied it per call.

6. **Vectorize the mask** — ~850 scalar accesses per tile became ~40 vector
   ops. Almost no time (I had over-estimated the loop by 14×), but exact, and
   it exposed that dispatch was now 85% of kernel1.

7. **Grid-stride kernel1, second attempt** — now worth it, because 3 and 6 had
   cut compute enough that dispatch dominated. Plus batching the `Decay` and
   gate row loops (~350 barriers per tile → ~25). Together 86.54 → 80.03 ms.

The re-profile that redirected everything: at T=1024 H=8 kernel2 is 79% launch
overhead, but at T=8192 H=64 it is 15%, because blockDim there is 64 rather
than 8. Optimizations chosen from the small-shape profile were aimed at the
wrong half.

## On single-run verification

Two intermittent faults in this work were hidden by running a check once:

- the `Gemm128` FIX_MTE2 race, which made the same shape give different errors
  run to run while the suite still passed;
- the grid-stride inter-unit race, where units sharing one TPipe were not
  drained between iterations — ~40% of runs, invisible in the one probe run I
  did before committing.

`tests/race_probe.py` and `tests/race_fields.py` need several runs to mean
anything. A clean single run is not evidence.

## What to do next

1. **Cheaper inter-unit sync in kernel1.** The grid-stride loop currently ends
   each unit with `PipeBarrier<PIPE_ALL>`, which costs roughly half of what
   grid-stride saves. The phases only need their own pipes drained, so
   per-queue barriers should recover most of that. Correctness first: this is
   the sync that was missing, so any change needs several race_probe runs.

2. **Kernel2's gemms are M=16.** `[16,128] × [128,128]` uses one row of
   fractals per MMAD. Batching several chunks' gemms is not possible (the state
   recurrence is sequential), but batching across *heads* is — heads are
   independent, and 64 of them share the same chunk index.

3. **Raise CHUNK from 16.** Chunk count drives kernel2's launch count directly
   and 16 is the cube's minimum fractal. CHUNK=64 would cut launches 4× and
   give each gemm 64 fractals of work. Large change: masks, buffer sizes, two
   more Neumann factors.

4. **Move FinishOut to the cube** (task #15). Would collapse 4 launches per
   chunk to 2, but forces `k_dec@state` through bf16 before a cancelling
   subtraction, so it trades precision — worth measuring, not worth assuming.

5. **Shrink the workspace.** `H × total_tiles × 608 KB`; every scratch slot is
   sized for a `[128,128]` fp32 state when only kernel2's state slots need it.
   18.6 GB at T=8192 H=64.

## A note on the numbers above

Every measurement here was taken after fixing a race in `Gemm128` (see
`tests/race_*.py`) under which only the last cube result before the kernel
ended was reliable. Before that fix the suite still passed, but the same shape
gave different errors on different runs, and `Mqk` was stably wrong. Timings
taken before it are not comparable.
