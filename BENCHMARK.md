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
| 8192 | 64 | 6.3 G | 61.03 | 1.6217 | **38×** |
| 8192 | 96 | 9.5 G | 85.70 | 2.6220 | **33×** |

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

1. **Batch the L2 normalization** — no change. Kept; chosen by reasoning, which
   is why it missed.
2. **Grid-stride kernel1, first attempt** — no change, reverted. Correct then:
   compute still dominated.
3. **Fuse the Neumann series** — kernel1 0.97 → 0.76 ms. 16 cube round trips,
   10 of them `X * I` copies, became 6.
4. **Fold BuildU into the previous FinishChunk** — 5 launches per chunk → 4.
5. **Stop reallocating the workspace** — 29.96 → 2.73 ms end-to-end, ~11×.
6. **Vectorize the mask** — exact, almost no time, but it exposed that dispatch
   had become 85% of kernel1.
7. **Grid-stride again + batch the row loops** — 86.54 → 80.03 ms.
8. **Size the scratch slots to what they hold** — workspace 18.6 → 6.3 GB, and
   80.03 → 76.58 ms.
9. **Declare kernel task types** — ~1.7%. Every phase runs on one core type,
   but without `KERNEL_TASK_TYPE_DEFAULT` each launch starts blocks on both.
10. **Fuse kernel2 into one kernel** — the big one. 77.00 → 61.23 ms, 47× → 38×.
11. **Fuse kernel1** — 0.3%, its four launches were per call, not per chunk.

Discarded after measuring (all interleaved, all no change):

- batching DecayState's 128-iteration scalar loop — the flushes were not the
  cost;
- casting the state from UB instead of re-reading 64 KB from GM — 2.0 GB of
  reads removed, worth nothing, because the data was L2-resident;
- narrowing the grid-stride barrier to `PIPE_MTE3` — also degraded race_probe
  from 6/6 to 5/6.

## The measurement that redirected everything

Launch overhead was estimated at 12–15% from a per-launch cost measured at
blockDim=1. At blockDim=64 it is four times that. Measured directly, with every
kernel2 phase stubbed so only launches remain, at T=4096 H=64:

| launches/chunk | ms |
|---:|---:|
| 4 | 36.6, 37.8 |
| 2 | 26.0, 26.9 |
| 1 | 21.0, 21.3 |
| kernel1 alone | 15.9 |

Perfectly linear, ~5.2 µs per launch — 21.6 ms, **54% of the pipeline**, while
all of kernel2's phase compute is 2.5 ms. `benchmarks/launch_bound.py` then
showed it is device-side: the host issues 1031 launches in 2.71 ms while the
device spends 37.15 ms. So issuing launches more cheaply could not help; only
issuing fewer.

## Cross-core sync, and two wrong conclusions

Every launch boundary in kernel2 existed to hand off between AIC and AIV, and
twice I concluded `CrossCoreSetFlag`/`WaitFlag` hang from a Python extension and
designed around it. Both were wrong. It needs
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` in the kernel body —
without a declared task type the runtime never provisions FFTS sync, so the
first wait blocks forever. Found only because `KERNEL_TASK_TYPE` turned up while
looking for a way to stop mix launches starting blocks on both core types.

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

1. **Kernel1 is now the larger half** (~16 ms of 30 at T=4096 H=64). It has had
   the launch and barrier work; what is left is the actual vector and cube work
   per tile.
2. **Kernel2's gemms are M=16.** `[16,128] × [128,128]` uses one row of
   fractals per MMAD. Heads are independent and 64 of them share a chunk index,
   so batching across heads is possible; the state recurrence is what cannot
   batch.
3. **Raise CHUNK from 16.** Fewer, larger chunks: 16 is the cube's minimum
   fractal. Large change — masks, buffer sizes, two more Neumann factors.
4. **Move FinishOut to the cube** (task #15). Now much less attractive: it saved
   launches, and launches are no longer the bottleneck.

## A note on the numbers above

Every measurement here was taken after fixing a race in `Gemm128` (see
`tests/race_*.py`) under which only the last cube result before the kernel
ended was reliable. Before that fix the suite still passed, but the same shape
gave different errors on different runs, and `Mqk` was stably wrong. Timings
taken before it are not comparable.
