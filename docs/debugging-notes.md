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
