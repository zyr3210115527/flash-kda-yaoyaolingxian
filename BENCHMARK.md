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

Full pipeline, and kernel1 alone (`FLASH_KDA_SKIP_K2=1`):

| T | H | workspace | per-call alloc | hoisted | kernel1 only | µs/token/head |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 154 M | 19.97 | 1.63 | 0.40 | 0.3973 |
| 1024 | 8 | 302 M | 29.88 | 3.18 | 0.76 | 0.3885 |
| 2048 | 8 | 599 M | 53.57 | 6.33 | 1.47 | 0.3861 |
| 2048 | 16 | 1198 M | 101.94 | 9.13 | 2.90 | 0.2785 |
| 4096 | 8 | 1193 M | 104.38 | 12.73 | 2.90 | 0.3885 |

Kernel2 is now the expensive half (~2.4 ms of 3.18 at T=1024 H=8), not
kernel1. That inverts the original conclusion, which was an artifact of the
allocation.

## Against the CUDA version

The CUDA implementation's own benchmark, on an H20:

| Shape | ms | µs/token/head |
|---|---:|---:|
| T=8192 H=64 D=128 fixed | 1.6217 | 0.0031 |
| T=8192 H=96 D=128 fixed | 2.6220 | 0.0033 |

Roughly **90× slower per token-head**, not the 790× the allocation-dominated
numbers suggested.

Different hardware (910B vs H20) and different shapes — our workspace does not
reach T=8192 — so treat this as a scale reference, not a like-for-like result.

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

## Where the time actually goes in kernel1

Phase by phase, cumulative, each phase stubbed out and added back:

| | ms |
|---|---:|
| launch only, all phases stubbed | 0.06 |
| + Prepare (AIV) | 0.28 |
| + LMqk (AIC, 2 gemms) | 0.30 |
| + Mask (AIV) | 0.47 |
| + Neumann (AIC, 16 gemms) | 0.98 |

Neumann was over half of kernel1, and the two vector phases most of the rest.
The cube gemms in LMqk are nearly free.

## What was tried

Measured, in order, with the outcome rather than the prediction:

1. **Batch the L2 normalization** — predicted the biggest win, delivered no
   measurable change. `NormalizeRow` ran 32 times per tile, each with a V_S/S_V
   round trip; batching it to 2 flushes was correct but the flushes were not
   the bottleneck. Kept because it is the same arithmetic in less code. The
   mistake was picking it by reasoning instead of measuring first.

2. **Grid-stride kernel1** — no measurable change, and reverted. blockDim was
   `total_tiles * H` (512 at T=1024 H=8) against 20 physical cube cores.
   Dispatching 512 blocks costs 0.55 ms with the phases stubbed versus 0.05 ms
   with 20, so the scheduling work is real — but total kernel1 time was 0.97 ms
   either way, because dispatch overlaps with compute. Reverted after it also
   turned out to expose a hazard: the phase code assumes one unit per core, and
   hoisting the per-unit `TPipe` out of the loop (which the pattern requires)
   corrupted `Mqk` outright.

3. **Fuse the Neumann series** — the one that worked. 0.97 → 0.76 ms for
   kernel1, about 20%. `ComputeNeumann` issued 16 cube round trips, and 10 of
   them were `X * I`: matrix copies done as 16×16×16 matmuls, a full
   GM→L1→L0→cube→L0C→GM trip with five barriers to move 512 bytes. They existed
   because `Gemm16` cannot write its destination in place. Replaced by a fused
   `dst = A*B1 + A*B2` (two Mmads, one L0C accumulator, one Fixpipe) plus
   ping-pong buffers, so each factor `P*(I + L^k)` is one trip instead of three.
   16 round trips became 6.

## What to do next

1. **Kernel2**, which is now ~75% of the pipeline and has never been profiled.
   The five-launches-per-chunk structure is the obvious suspect, but that is
   exactly the kind of guess that was wrong twice above — measure first.
2. **Vectorize `MaskAndBuild`.** It builds a 16×16 mask with a scalar double
   loop, 256+ `GetValue`/`SetValue`. The Mask phase is 0.17 ms.
3. **Fuse the decay loop** — 16 iterations over a `[16,128]` tile that could be
   whole-tile vector ops. Part of the 0.22 ms Prepare phase.
4. **Shrink the workspace.** `H × total_tiles × 608 KB`, because every scratch
   slot is sized for a `[128,128]` fp32 state when only kernel2's state slots
   need that, and those are per (sequence, head) not per (tile, head). At
   T=8192 H=64 the current layout wants ~20 GB. This is what blocks
   benchmarking at the CUDA version's shapes, and it also dominates real-world
   cost: the per-call allocation above is 88–98% of end-to-end time for any
   caller that does not hoist it.

## A note on the numbers above

Every measurement here was taken after fixing a race in `Gemm128` (see
`tests/race_*.py`) under which only the last cube result before the kernel
ended was reliable. Before that fix the suite still passed, but the same shape
gave different errors on different runs, and `Mqk` was stably wrong. Timings
taken before it are not comparable.
