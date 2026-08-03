# FlashKDA-Ascend — status

## Where things stand

**The forward pass works, matches the CUDA reference, and has been optimised.**

12/12 shape and feature cases pass against `torch_ref` at the bf16 noise floor,
bit-identical across repeated runs, on all four fallback paths:

```
T=16  H=1  one chunk      6.157e-03    B=2   T=32 H=2        8.516e-03
T=64  H=4  baseline       6.537e-03    state bf16  out 7.571e-03  state 4.662e-03
T=48  H=2                 9.727e-03    state fp32  out 7.571e-03  state 4.595e-03
T=128 H=8                 7.178e-03    varlen [16,24,8] H=1  5.956e-03
T=80  H=1                 6.487e-03    varlen [32,32]   H=2  8.516e-03
T=40  H=2  tail 8 rows    7.312e-03
T=24  H=4  tail 8 rows    7.270e-03
```

Sequence lengths 16–8192, head counts 1–96, batching, ragged varlen, tail
chunks, both state dtypes.

Determinism is checked separately and repeatedly: `tests/race_probe.py` runs the
same input eight times and diffs bitwise, and it takes six runs of that to mean
anything — two intermittent races in this project were hidden by a single clean
run. `tests/workspace_poison.py` fills the workspace with NaN to prove nothing
is read before it is written.

## Performance

32.58 ms at T=8192 H=64, **20×** the CUDA implementation on an H20 (1.62 ms).
Full accounting in [BENCHMARK.md](BENCHMARK.md); the short version:

| | |
|---|---|
| Correct but unoptimised, first measurement | 790× |
| Same code, measured properly | ~94× |
| After 16 optimisations over five days | **20×** |

Most of 790 → 94 was a broken harness, not a fix. What follows is real.

## Running it

Get a card (see [`docs/bringup.md`](docs/bringup.md)), then:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/usr/local/python3.11.14/bin:$PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd FlashKDA-Ascend/build && make -j8 && cp _C*.so ../flash_kda/
cd .. && PYTHONPATH=$PWD:$PWD/tests python3 tests/test_shapes.py
```

| Test | What it covers |
|---|---|
| `tests/test_all_parse.py` | that every test and benchmark parses. No hardware, under a second. Four probes sat broken for days because the suite was green and nobody ran them. |
| `tests/test_shapes.py` | the 12 shape and feature cases |
| `tests/race_probe.py` | determinism — bitwise diff over repeats. Run it 6+ times. |
| `tests/race_fields.py` | which workspace field first goes non-deterministic |
| `tests/workspace_poison.py` | that nothing reads the workspace before writing it |
| `tests/test_neumann_strength.py` | the Neumann series at ‖L‖ large enough to matter |
| `tests/bf16_update_drift.py` | whether a precision change accumulates along the sequence |
| `tests/validate_ref.py` | the CPU oracle itself, against independent float64. No hardware. |
| `tests/check_prepare.py` | kernel1's workspace outputs, field by field |
| `benchmarks/ab_compare.py` | interleaved A/B with medians — resolves ~1% |
| `benchmarks/profile_pipes.py` | hardware pipe counters (cube MAC, mem-in, vector, scalar) |
| `benchmarks/bench_cuda_shapes.py` | the reference's own shapes |

Probes read the workspace layout from the extension (`_C.CHUNK`,
`_C.WS_SLOT_OFF`, …) rather than hardcoding offsets — those offsets have moved
twice, and a stale copy reads the wrong tile while looking entirely plausible.

## Design

Both kernels are now **single fused launches** that hand off between the cube
and vector cores with cross-core flags, and both vector cores participate.

```
kernel1 (grid-strided over tile×head)   kernel2 (one unit per sequence×head)
  AIV  prepare, both subblocks            AIV  InitState
  AIC  L = k_dec @ k_inv^T, Mqk           per chunk:
  AIV  tril mask, beta, (I - L)             AIV  FinishOut
  AIC  Neumann inverse, entirely in L1      AIC  PostGemms(c) then PreGemms(c+1)
                                            AIV  StoreOut ∥ DecayState (8/20 split)
                                          AIV  StoreFinalState
```

Every phase declares its core type with `if constexpr (g_coreType != ...)`, and
every entry declares a `KERNEL_TASK_TYPE` — without which the runtime never
provisions the cross-core sync resources and the first wait hangs forever.

Inter-phase values that cross between core types travel through GM, because on
Atlas A2 the AIC cannot reach UB. Values that stay within one core type do not:
the Neumann chain lives in L1 and the recurrent state lives in UB.

## Unresolved

- **INV's diagonal departs from 1 at ‖L‖ > 0.3.** A product of unit-lower-
  triangular factors has a unit diagonal for any input, so this is structural,
  not tolerance. Not reproducible at usable inputs — every regime that produces
  ‖L‖ that large also drives `k_decayed` to underflow — but it is the leading
  suspect for the CHUNK=32 failure. `tests/test_neumann_strength.py`.
- **A fused m=32 `PreGemms`** measures 4% faster and 1/12 correct. Four
  hypotheses eliminated; see `docs/debugging-notes.md`.
- **`MIX_AIC_1_2` with the two AIV subblocks in different phases** is not
  expressible — the arming rule is per-flagId but collective over the core's
  AIV group, so no subblock can arm a flag alone. Settled by three probes.

## Not done

- **Backward pass.** Forward only.
- **Ascend 950 (`CATLASS_ARCH=3510`).** Builds; only 910B has been exercised.
- **Larger D.** `D == 128` is asserted host-side.
- **CHUNK > 16.** Now a real knob (the Neumann length derives from it), but
  bounded by `CHUNK · |lower_bound| ≤ ~88` — a model constraint the CUDA
  reference shares, not something the kernel can choose away.
- **The hand-rolled layout descriptors.** `Nd2NzParams`, `LoadData2DParams` and
  `FixpipeParamsV220` are still written by hand. Every layout bug in this
  project was a hand-derived stride; the catlass tile classes would be the
  durable cleanup, and the 950 port will want it anyway.
