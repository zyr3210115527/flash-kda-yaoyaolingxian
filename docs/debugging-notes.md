# FlashKDA-Ascend — status

## Where things stand

**The forward pass works and matches the CUDA reference.**

12/12 shape and feature cases pass against `torch_ref` at the bf16 noise floor
(~1e-2), and 10/10 repeated runs pass with no faults or hangs:

```
T=16  H=1  one chunk      6.157e-03    B=2   T=32 H=2        8.516e-03
T=64  H=4  baseline       7.273e-03    state bf16  out 7.571e-03  state 4.662e-03
T=48  H=2                 9.727e-03    state fp32  out 7.571e-03  state 4.595e-03
T=128 H=8                 8.062e-03    varlen [16,24,8] H=1  5.956e-03
T=80  H=1                 6.487e-03    varlen [32,32]   H=2  1.249e-02
T=40  H=2  tail 8 rows    7.312e-03
T=24  H=4  tail 8 rows    7.270e-03
```

Sequence lengths 16-128, head counts 1-8, batching, ragged varlen, tail chunks,
and both state dtypes.

Kernel1's intermediates are additionally checked field by field against float64:

```
k_decayed 1.478e-03   q_decayed 1.674e-03   k_inv 1.785e-03
k_restored 1.734e-03  g_total 8.962e-07     Mqk 3.851e-03   INV 3.352e-04
```

The CPU oracle itself is verified: `tests/validate_ref.py` checks `torch_ref`
against an independently written float64 delta-rule loop, 5/5.

## Running it

Get a card (see `bringup.md`), then:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/usr/local/python3.11.14/bin:$PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /user/zhouyiran/flashkda/FlashKDA-Ascend/build && make -j8 && cp _C*.so ../flash_kda/
cd .. && PYTHONPATH=$PWD:$PWD/tests python3 tests/test_shapes.py
```

| Test | What it covers |
|---|---|
| `tests/validate_ref.py` | the CPU oracle, against an independent float64 implementation. No hardware needed. |
| `tests/check_prepare.py` | kernel1's seven workspace outputs against float64 |
| `tests/dump_k2_chunk0.py` | kernel2's chunk-0 intermediates, using the device's own Mqk/INV |
| `tests/per_chunk_err.py` | error broken down per chunk and per head |
| `tests/test_shapes.py` | the 12 shape and feature cases |
| `test_npu_nocompute.py` | single end-to-end number |

`FLASH_KDA_SKIP_K2=1` runs kernel1 alone. `_C.noop()`, `_C.aiv_only()` and
`_C.aic_only()` are launch-path probes; `tests/sync_probe.cpp` and
`tests/aic_resource_probe.cpp` are standalone comparisons.

## Design

Both kernels are split into single-core-type launches ordered by the stream,
rather than one kernel with AIC/AIV cross-core handshakes.

```
kernel1 (per tile, per head)          kernel2 (per sequence, per head)
  AIV  prepare                          AIV  InitState
  AIC  L = k_dec @ k_inv^T, Mqk         per chunk:
  AIV  tril mask, beta, (I - L)           AIV  BuildU
  AIC  Neumann inverse                    AIC  PreGemms
                                          AIV  FinishOut
                                          AIC  PostGemms
                                          AIV  FinishChunk
                                        AIV  StoreFinalState
```

Every phase states its core type with `if constexpr (g_coreType != ...)`: the
compiler emits both a `_mix_aic` and a `_mix_aiv` half for each kernel, and an
unguarded body runs on both.

All inter-phase values travel through GM workspace, which is what made the split
cheap -- on Atlas A2 the AIC cannot reach UB and L0C drains only to L1 or GM.

## Things that cost real time, worth knowing

**Kernel entries must take a single argument.** Five of kernel2's entries took
`(FwdParams, int32_t)` and hung; the two taking a lone `FwdParams` ran. Moving
the chunk index into `FwdParams` fixed it. This also invalidated an earlier
conclusion that cross-core sync does not work from a Python extension -- those
probes were two-argument too, so the experiment was confounded. Whether
cross-core sync works with a single-argument entry is untested; the split is a
reasonable design regardless, but it was not forced by what I thought forced it.

**`DataCopyPad` pads every block to 32 bytes.** Gathering one value per token
with `blockLen = sizeof(bf16)` does not land them contiguously -- they land 16
elements apart with garbage between. This corrupted the beta load in both
kernels (`sigmoid(beta)` at rel 6.1e-01, with every downstream quantity
inheriting it) and the state transpose. Both now copy whole blocks and select
with scalar reads.

**`M_MTE1` after every `Mmad`.** The cube reads L0A/L0B and the next GEMM's
`LoadData` overwrites them. Without the barrier the device reports
`"L0B read/write conflict in the MTE (same address)"` intermittently -- two runs
in six.

**`Nd2Nz` describes a ColumnMajor source by its columns.** `nValue` is the
column count, `dValue` the column length. Reversed, it read to element 16272 of
a 2048-element buffer.

**The Neumann iteration squares L, not (I - L).** `(I-L)^2 = I - 2L + L^2`. The
symptom was `2^3 = 8` on the inverse's diagonal.

**No scalar `exp` on the aicore, and scalar bf16 casts are rejected.** Anything
exponential goes through the vector unit; widen bf16 with `Cast` before reading
it scalar.

**Exact-fit L0 allocations faulted; architectural sizes fixed it.** But the
opposite held in kernel2's L1 map, where widening slots to 64 KB broke it. Do
not generalise either way without measuring.

Two of my own testing bugs, since they produced convincing wrong answers:
`struct` has no bf16 code and `'e'` is IEEE fp16, so reading bf16 workspace
fields with `'e'` made correct data look wrong by 20-500x; and the sync script
built into `build/` without installing the `.so`, so two rounds of "identical
results" were a stale binary.

## Not done

- **Performance.** Nothing has been tuned or measured. Kernel2 issues five
  launches per chunk, the state transpose is a scalar double loop, and there is
  no double buffering anywhere. Correctness first was the right order, but this
  is not a fast kernel yet.
- **The hand-rolled layout descriptors.** `Nd2NzParams`, `LoadData2DParams` and
  `FixpipeParamsV220` are still written by hand. Every layout bug in this
  project was a hand-derived stride; the catlass tile classes (`CopyGmToL1`,
  `CopyL1ToL0A`, `CopyL1ToL0B`, `CopyL0CToGm`) derive them from layout objects
  and would be the durable cleanup.
- **Backward pass.** Not started; this is forward only.
- **Ascend 950 (`CATLASS_ARCH=3510`).** Only 910B has been exercised.
- **Larger D.** `D == 128` is asserted host-side.

## Why CHUNK cannot simply be raised (measured 2026-08-01)

Hardware counters put the cube at 2–3% MAC utilization with memory-in as its
busiest pipe: it is fetching operands, not computing, because CHUNK=16 makes
every matmul one fractal row. Raising CHUNK is the only change that lifts that,
so it was worth attempting.

The plumbing side went fine, and is neutral at CHUNK=16 (12/12 with identical
errors, 21.72 ms unchanged). CHUNK is now a real knob: the Neumann series
derives its length from `kLog2Chunk` instead of four hardcoded factors, the
16×16 tiles in K1L1/K1Ub are sized `CHUNK*CHUNK`, `repeatTimes` is
`(CHUNK/16)^2` rather than 1 in five places, kernel2's `kNarrow` overlays the
front buffers so UB still fits, and tests read `CHUNK` and the workspace offsets
from the extension instead of each keeping a copy.

CHUNK=32 builds and runs and produces NaN everywhere. Two separate causes.

**1. The decay representation overflows fp32 — not a bug.**

`k_inv = k * exp(-cumsum(gate))`, and the gate is clamped at `lower_bound` per
token, so the exponent reaches `|lower_bound| * CHUNK`. Sweeping the clamp at
CHUNK=32 tracks it exactly:

| lower_bound | exponent | k_inv absmax |
|---|---|---|
| −5.0 | e^160 | inf |
| −2.0 | e^64 | 1.98e+20 |
| −1.0 | e^32 | 5.27e+09 |

At CHUNK=16 with the usual −5.0 that is e^80 = 5.5e34, just inside fp32's
3.4e38. At CHUNK=32 it is e^160, which is not.

So **CHUNK=16 in the CUDA reference is not an arbitrary tile choice** — it is
about the largest chunk for which this formulation stays in fp32 range at
lower_bound=−5. Raising it needs the decay re-based within the chunk (subtract
a per-chunk reference exponent, fold it into the state update) rather than
just bigger tiles. That changes the math, so both oracles follow.

**2. A layout bug on top of it.**

INV is NaN even at lower_bound=−1.0, where k_inv is 5.27e9 and every input is
finite. Something in the Neumann chain is still 16×16-specific beyond
`repeatTimes` — most likely the Nd2Nz or Fixpipe strides in the Gemm16 family,
which assume a single fractal where `srcStride`/`dstStride` of CHUNK happen to
coincide with the zN layout. `LoadBt`'s `ldb.srcStride = 1` carries a comment
that 8 "made a working single tile hang", which is the same class of
hardcoding, and is what task #8 (replace hand-rolled fractal strides with
catlass tile classes) is about.

`tests/chunk32_probe.py` dumps each kernel1 field for a one-tile case and says
which first goes bad, using the exported offsets.

Reverted to CHUNK=16. The parameterization is kept: it makes the next attempt a
matter of fixing strides and re-basing the decay, rather than untangling
constants first.

## MIX_AIC_1_2: six attempts, and why it is parked (2026-08-02)

Motivation, from the hardware counters: AIC and AIV **alternate rather than
overlap** — for kernel2 at T=4096 H=64, AIC's busy pipes total 4.78 ms and AIV
is busy 6.65 ms against a 13.2 ms kernel, i.e. their sum. A2 pairs each cube
core with two vector cores, and the profiler confirms the runtime hands out the
1:2 default to an unannotated kernel (`FlashKdaK2InitState`: Block Dim 64, Mix
Block Dim 128) while our `MIX_AIC_1_1` kernels get 64. So declaring 1_1 halves
the vector cores available. Using the second one should cut the AIV half.

What was tried, in order:

1. **Switch the macro to `MIX_AIC_1_2`.** Deadlock (>10 min, processes killed).
   The AIV phases open with `if (GetSubBlockIdx() != 0) return;`, so subblock 1
   leaves without participating and the AIC waits for a partner that never
   arrives.
2. **Only subblock 0 handshakes; the pair rendezvous with
   `CrossCoreBarrier<0x1>` (AIV_INTER_SUBBLOCK_BARRIER).** Also deadlocks. So
   one subblock cannot signal on behalf of both.
3. **Both subblocks handshake symmetrically, in an isolated probe.** Works —
   three runs each at blockDim 1, 8, 64, 64, 256. This is the rule:
   `CrossCoreSetFlag<0x2>` is not satisfied until *every* AIV in the group has
   set the flag.
4. **Real kernel1, work split across subblocks** (`start = blockIdx`,
   `stride = blockNum * subBlockNum`). Builds, runs, no deadlock — but NaN,
   1/12.
5. **Real kernel1, 1_2 handshake but the original single-subblock work split.**
   Also 1/12. So the mode change alone breaks it; the split is not the fault.
6. **AIC waits once per AIV** (`kAivPerAic` times per phase), on the theory
   that two sets against one wait leave a pending flag that releases the next
   phase early. Deadlock.

Conclusion: the flag semantics that hold in an isolated probe do not carry over
once phases have real data dependencies, and I could not find the correct
pairing in six attempts. The isolated probe passing (3) is exactly why it is
worth recording — it is a false positive for the real kernel, and repeating it
proves nothing.

Measured indexing under 1_2, which is correct and reusable: on the AIV side
`GetBlockNum()` returns the **cube** block count, `GetSubBlockNum()` is 2, and
`blockIdx` runs over twice the cube blocks with `subBlockIdx = blockIdx % 2`.
Obtained by writing the values to GM — device `printf` needs `ENABLE_PRINTF()`
*and* a host-side dump buffer that is not configured here, so it stays silent.

Parked at `MIX_AIC_1_1`, 21.63 ms, 12/12, 6/6 clean race probes. The prize is
real (perfect overlap would put kernel2 near 6.6 ms rather than 13.2), but it
needs the actual flag protocol for 1:2, which likely means a documentation
source rather than more black-box probing.

## PreGemms at m=32: faster and wrong, parked with the diagnosis half-done

The counters after the subblock split put `cube mem-in` as the busiest pipe in
both kernels (0.351 kernel1, 0.296 kernel2) against a cube MAC ratio of
0.025–0.044. The cube is starved on operands, and the reason is m=16: every
Mmad is one fractal row.

`PreGemms` is the natural place to attack it. It runs `k_dec @ state` and
`q_dec @ state` — same B operand, and `k_decayed`/`q_decayed` are already
adjacent in the workspace. So `[k_dec; q_dec]` loads as one `[2*CHUNK, D]` A
operand and the pair becomes a single m=32 gemm. Same MAC work, double the m.

Measured: **15.63 ms against 16.26, about 4%** — and 1/12 correct.

What was checked and was not the cause:
- `LoadNd2Nz` handles m=32 (it takes `rows` and sets `nValue`/`dstNzC0Stride`
  from it);
- `repeatTimes = (m/16)*(k/16)` scales;
- `Drain`'s Fixpipe uses `mSize = m`, `srcStride = m`;
- the L1 A buffer overrunning into `kB` — `kA` was sized `CHUNK*D*2` and the
  fused load writes twice that. Doubling it changed nothing (the error value
  stayed at exactly 1.727), so it was not the fault, though the sizing was
  genuinely wrong and the fix is kept.

Two further attempts, both wrong, both worth recording:

- **A-operand `srcStride` derived from m.** kernel1's `Gemm128` computes it as
  `m*16/256`, kernel2's `Gemm` hardcodes 1, and 1 is only right at m=16 — so
  this looked like the answer. It made things *worse*, error 1.727 → 37.45,
  which says the L1 zN layout at m=32 is in fact what `srcStride = 1`
  describes, and the fault is elsewhere.
- **A host-side diff of slots 4 and 6.** Ran it against the full pipeline
  rather than `FLASH_KDA_SKIP_K2=1` (those slots are kernel2's, so the flag
  leaves them empty) — and got all zeros *including for the host reference*,
  because at that probe's inputs `k_decayed` is itself zero. The probe was
  measuring nothing. A useful reminder that a reference agreeing with the
  kernel at 0.0000 is not agreement.

So the diagnosis is still open, and the honest summary is that four separate
hypotheses have been eliminated without finding it. What is needed is a probe
that runs kernel2 with non-degenerate inputs and reads slots 4 and 6 before the
next chunk overwrites them — one chunk, realistic k/g, and a host reference
that is verified non-zero first.

**Kept:** the slot reordering that makes 4 and 6 adjacent
(`WorkspaceSizes::NarrowIndex`). It is neutral on time (16.26 either way) and
12/12, and it is the prerequisite for the fusion — without it the two outputs
are separated by slot 5 and no fused gemm is possible. Slot 5 is kernel1's
plain L, live only inside `ComputeNeumann`, so nothing observes the move.

**Kept:** the `kA` L1 sizing fix, since a fused load of that shape is the
intended future and the old size was wrong for it.

The 4% is worth returning to, and the m=16 problem it attacks is the same one
CHUNK is stuck on — a fused m=32 gemm here would be a cheaper partial win than
raising CHUNK, since it needs no decay re-basing.

## The CHUNK ceiling is CHUNK · |lower_bound| ≤ ~88, and it is shared with the reference

Kimi's round-2 notes suggested the CUDA/FLA reference avoids the
`exp(−cumsum)` overflow entirely by never forming the reciprocal — binding
a = b = k so the 1/Γ factors cancel. Checked against the reference in this
repo, and it does not transfer, for two reasons.

**The gate is per-channel.** `FlashKDA/csrc/smxx/fwd_kernel1.cuh` cumsums down
rows *within a column* and stores `[CHUNK, D]`:

    for (int row = 0; row < CHUNK; ++row) { sum += g_val;
                                            g_smem[row * D + col] = sum; }

So `exp(g_i − g_j)` is `[C, C, D]`, not `[C, C]`, and cannot be pulled out of
the `k_i · k_j` dot product as a scalar — the exponent varies along the
contracted dimension. The identity `A[i][j] = (k_i·k_j)·exp2(g_i − g_j)` holds
only for a scalar per-head gate.

**The reference forms `k_inv` explicitly**, same file:

    BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g));
    r_ki(0, v) = k * inv_cumsum;
    mma_m16n16_bf16bf16fp16_1warp(k_decayed, k_inv, L_fp16, compute_tid);

Identical structure to this port, and `constexpr int CHUNK = 16` there too. The
FLA Triton kernel at chunk 64 is a different implementation.

**The shared bound.** The reference keeps `k_inv` in bf16, which has fp32's
8-bit exponent, so the ceiling is the same. Per-step decay is bounded by
`|gate_scale|` since `gate = gate_scale · sigmoid(...)`. And the log bases
cancel rather than differing — reference uses
`gate_scale = lower_bound · log2(e)` with `ex2`, this port uses raw
`lower_bound` with natural `Exp`. So for both:

    CHUNK · |lower_bound| <= ~88

CHUNK=16 at lower_bound=-5 is 80, inside 88 but only just — which is exactly
the measured behaviour (-5 overflows at CHUNK=32, -2 gives 1.98e20, -1 gives
5.27e9). CHUNK=32 needs |lower_bound| <= 2.75; CHUNK=64 needs <= 1.375.

So raising CHUNK is not a formulation choice available in the kernel. It trades
against a model parameter, and a larger-CHUNK build should range-check
`lower_bound` at launch rather than assume.

## The full mode-0x2 arming rule (measured 2026-08-03)

Kimi's §1(a) question: when two AIV subblocks are in different phases, can
per-subblock flagIds express it? Three probes settle the rule.

1. **sub0 sets flag 3, sub1 sets flag 4, AIC waits 3 then 4** — deadlock.
2. **both subblocks set flag 3 AND flag 4, AIC waits 3 then 4** — completes,
   and is stable at 8, 64 and 512 iterations.
3. (earlier) **both set one shared flag, AIC waits twice** — deadlock.

So arming is **per flagId, but collective over the AI Core's AIV group**:

> An AIC wait on flagId X is satisfied only once *every* AIV in the core's
> group has set X. One wait consumes the whole group's set of X.

Both halves matter and they are easy to conflate. Multiple flagIds *are*
distinguishable — probe 2 shows two flags carrying two separate events works —
but a single subblock can never arm a flag alone. So:

- **Possible**: several flagIds, each meaning a different event, each set by
  both subblocks. Probe 2, stable to 512 iterations.
- **Not possible**: using flagId identity to tell the subblocks apart, e.g.
  "flag 3 means sub0 finished phase c+1, flag 4 means sub1 finished phase c".
  That is exactly probe 1, and it hangs.

Consequence for the skewed pipeline: Kimi's design (a) — per-subblock flagIds
letting the two AIVs sit in different phases — is **not expressible**. Design
(b) is the whole space: keep the handshakes phase-symmetric and reached by both
subblocks, and skew the *work* between them. That is what the 8/20 uneven
StoreOut/DecayState split already does, and it generalises.

Which is a constraint on the design but not a blocker: the AIC-side skew Kimi
described (AIC runs PreGemms(c+1) while the AIVs finish chunk c's vector work)
never needs the two AIVs in different phases — only the AIC ahead of both. That
remains expressible with symmetric flags, and multiple flagIds are available to
separate the extra handshake it needs.

Probe: `_C.persub_flag(n, h, iters)`, `FlashKdaPerSubFlag` in fwd_kernel2.asc.

## Halving kernel2's handshakes: correct, neutral, and built on bad arithmetic

Restructured kernel2's chunk loop so the AIC does `PostGemms(c)` and
`PreGemms(c+1)` in one turn instead of two, cutting the handshake from 4 round
trips per chunk to 2. It is legal — `PostGemms(c)` needs `u(c)` from
`FinishOut(c)`, `PreGemms(c+1)` needs slots 3 and 2 of tile c+1 which the AIV's
end-of-chunk phase writes, and slots are keyed by `tileIdx` so c and c+1 do not
alias. 12/12 and 6/6 clean races.

**Worth 0.00 ms.** 16.26 before, 16.26 after, and the pipe ratios are unchanged
(mem-in 0.299, MAC 0.025).

The reason it was expected to help was an arithmetic mistake worth recording.
I had computed "AIC busy 4.8 ms + AIV busy 3.3 ms = 8.1 ms against a 10.3 ms
kernel, so 2.2 ms is handshake latency" — but `aic_*_ratio` are fractions of
`aicore_time` and `aiv_*_ratio` are fractions of `aiv_time`, which are
different denominators. Summing them is not meaningful. Done correctly the
unaccounted time is ~0.64 ms, not 2.2, so there was never 2.2 ms of handshake
to remove and the prediction was of a quantity that does not exist.

Kept anyway: it is strictly fewer synchronisation points for the same work, it
is the structure a future skew would need, and it costs nothing. But it is
listed here as a null result, not a win.

Related and also settled: an AIC-side skew deeper than this is **not available**.
`FinishOut(c+1)` reads slot 4, which is `PreGemms(c+1)`'s output, so the AIV
genuinely cannot start chunk c+1's vector work before the cube has produced it.
The AIC/AIV alternation in this kernel is a true data dependency, not a
handshake artifact — which is the useful thing the exercise established.

## The cube is latency-bound on mem-in, and the FIX_MTE2 barrier is why — but it is load-bearing

Post-subblock-split, `cube mem-in` is the busiest pipe in kernel2 (0.299
against a MAC ratio of 0.025). Quantifying what it actually moves:

    PreGemms A: k_dec + q_dec     8.0 KB
    PreGemms B: state bf16       32.0 KB
    PostGemms: INV, u             4.5 KB
    PostGemms: Mqk, uu            4.5 KB
    GemmAt: kres, uu              8.0 KB
    per chunk                    57.0 KB   -> 0.89 GB total

0.89 GB in the 3.11 ms that pipe is busy is **287 GB/s against ~800 available**.
So it is *latency*-bound, not bandwidth-bound — the pipe is busy 30% of the
time moving very little, which means small transfers that do not overlap.

Why they do not overlap: every `Gemm` opens with a `FIX_MTE2` barrier, which
blocks this gemm's GM loads behind the *previous* gemm's Fixpipe.

Two attempts to relax it:

- **Move it to just before `Drain`**, where the write actually happens.
  15.89 ms against 16.26, −2.3%, and **0/12**.
- **Make it conditional**, skipping it in `PreGemms` where the two gemms write
  different slots and read neither (`PostGemms` genuinely needs it: `Gemm(INV,
  u)` writes `uu` and both later gemms read it). 15.76 ms, −3.1%, and
  **11/12** — every case's error degraded slightly (baseline 6.54e-3 → 7.69e-3)
  with T=48 tipping past tolerance at 3.5e-2.

That 11/12 is the informative one. A uniform small degradation with one case
crossing the line is a *partially ordered read*, not a clean dependency break —
so `PreGemms` does have a hazard the barrier covers, beyond the
gemm-reads-previous-gemm's-output one I enumerated. The state as B operand is
written by the AIV, so it is not that either; something else in the load path
is ordered by this barrier.

Both reverted. The 3% is real and available if the actual hazard is identified,
but not worth taking blind — and the error degrading *everywhere* while only
one shape fails is exactly the pattern that would pass a less thorough sweep.

## The FIX_MTE2 in PreGemms was covering the wrong hazard (and the 3% was a race)

Follow-up to the entry above, which left "3% is available if the real hazard is
found" open. It is not — but 1.2% is, and the reasoning that got there corrects
two things I had written down.

Bisecting the two `PreGemms` gemms separately, four runs of the shape sweep and
three race probes each:

| barrier skipped | shape sweep | race probe |
|---|---|---|
| none (baseline) | 12 12 12 12 | 6 6 6 |
| first gemm only | 12 12 12 12 | 6 6 6 |
| **second gemm only** | **11 12 11 12 11** | **0 0 0** |
| both | 11 12 11 12 12 | 0 0 0 |

So the whole hazard is in the **second** gemm, and it is a **race** — not the
deterministic dependency the earlier 11/12 looked like. That also explains the
signature that puzzled me: every case's error degrading slightly with one shape
tipping over tolerance is exactly what a race looks like, and I had read it as a
partially ordered read. A single sweep is not evidence here; the same config
gave 11/12 and 12/12 on alternating runs.

Skipping the first gemm's barrier alone is safe and worth **0.00 ms**, so the
3% I measured earlier came entirely from removing the *unsafe* one. There was
never a free 3%.

**Why the second gemm and not the first.** It is called with `loadB=false` — it
reuses the state already in L1 — so all it loads is A, into the *same* `l1A` the
first gemm is still feeding to L0A via MTE1. That is an **L1 buffer-reuse
hazard**, and `FIX_MTE2` was only covering it *transitively*: Fixpipe comes
after Mmad comes after MTE1, so waiting on the Fixpipe happens to wait on the
MTE1 too.

Naming the real dependency instead — `MTE1_MTE2` — is correct (12/12 × 4, 6/6
clean race probes × 3) and cheaper: **16.06 ms against 16.26**, 1.2%, measured
interleaved across three pairs with non-overlapping ranges.

Narrowing both `PreGemms` gemms measures the same as narrowing only the second
(16.06 vs 16.06), and both are taken, because both have the same dependency and
neither reads a prior Fixpipe's output. `PostGemms` keeps `FIX_MTE2`: there
`Gemm(INV, u)` writes `uu` and both later gemms read it, so the Fixpipe really
does have to land first.

The general shape, worth remembering: **a barrier that is correct can still be
the wrong barrier.** `FIX_MTE2` was doing two jobs here, one of them by
accident, and the accidental one was the only one that mattered in `PreGemms`.

### Where the narrowing pays, and where it does not

Having found that `MTE1_MTE2` is the right barrier for a gemm whose inputs do
not come from a preceding Fixpipe, the obvious next move is to apply it
everywhere that holds. There is exactly one other such gemm: `PostGemms`' first,
`Gemm(INV, u) -> uu`, which reads kernel1's `INV` and the AIV's `u`. (The two
after it read `uu` and genuinely need `FIX_MTE2`.)

Narrowed it. Correct — 12/12 four times, 6/6 clean race probes four times — and
worth **0.00 ms**: 16.06 against 16.06.

**Reverted**, because a barrier change that buys nothing is pure risk surface in
a kernel that has already produced three races, and `FIX_MTE2` is the stronger
of the two.

But the null result explains the positive one, which makes it worth keeping in
writing. A `FIX_MTE2` only *costs* anything when the previous Fixpipe is still
in flight when the next gemm wants to load. That is true for `PreGemms`' two
gemms, which run back to back on the cube with nothing between them. It is false
for `PostGemms`' first gemm, which is separated from the preceding gemm by an
entire AIV phase and two cross-core handshakes — by the time the cube gets
control again, the Fixpipe drained long ago and the barrier is free.

So the rule for where this optimisation is available: **back-to-back gemms on
the same core, not gemms separated by a handoff.** That predicts there is
nothing further to collect here, and it is the reason to stop looking.
