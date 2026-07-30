# FlashKDA-Ascend — session handoff, 2026-07-30 evening

## Where things stand

Both kernels are **rewritten**, the extension **builds and links**, and it
**launches on a real 910B3**. Kernel1 has been restructured to four
single-core-type launches and its two AIV phases now **run correctly on device**.
The two AIC (cube) phases still hang, so the forward pass does not complete end
to end yet.

An AIV-only probe kernel produced **numerically exact** output on the card
(`exp(0)=1.000000`, `exp(1.27)=3.560853`), so the vector path, the workspace, the
launch plumbing and the build are all genuinely working.

## The bug is in the cube path, and it is localized

Two findings, both from experiments on hardware.

**1. Cross-core sync does not work from a Python extension.** Seven variants,
identical two-round handshake:

| Variant | Where | Result |
|---|---|---|
| `sync_probe`, blockDim 1 / 2 / 4 | standalone binary | completes |
| `sync_probe` + `Resource<ArchTag>` member | standalone binary | completes |
| `sync_probe` + `PipeBarrier<PIPE_ALL>` | standalone binary | completes |
| `noop` kernel, no handshake | in the .so | completes |
| handshake only, `FwdParams` by value | in the .so | **hangs** |
| handshake only, two scalar args | in the .so | **hangs** |
| kernel1 with all compute removed | in the .so | **hangs** |

The handshake works standalone; a kernel without one works in the extension.
Only the combination fails.

Ruled out by experiment: blockDim; `Catlass::Arch::Resource` and its `TPipe`;
`PipeBarrier<PIPE_ALL>`; torch's stream (a private stream with an explicit
synchronize hangs too); kernel argument size; `aclrtGetHardwareSyncAddr`
(returns `rc=0` and a plausible address inside the extension); and a missing
mix-kernel task type — no `KERNEL_TASK_TYPE` / `SetKernelTaskType` exists in
catlass or in this CANN 8.5.0's `tikcpp` headers.

**So kernel1 was restructured into four separate launches** — AIV prepare, AIC
L/Mqk, AIV mask, AIC Neumann — ordered by the stream instead of by flags. Every
inter-phase value already travelled through GM workspace, so this was cheap.
With all four phase bodies stubbed out, the four launches complete.

**2. With the split in place, the hang is isolated to the cube phases.** Running
one phase at a time:

| Phase | Core | Result |
|---|---|---|
| `RunPrepare` — normalize, gate, cumsum, decay, store | AIV | **completes** |
| `RunLMqk` — `L = k_dec @ k_inv^T`, `Mqk = q_dec @ k_inv^T` | AIC | **hangs** |
| `RunMask` — tril mask, beta sigmoid, build `(I - L)` | AIV | **completes** |
| `RunNeumann` — the inverse | AIC | **hangs** |

Both AIV phases work. Both AIC phases hang. The fault is in the shared cube
path: `Nd2NzParams` → `LoadData2DParams` → `Mmad` → `FixpipeParamsV220`.

## What to do next

This is the hand-derived stride code, and it should be replaced rather than
patched — hand-computing fractal strides is the habit that produced the draft
this rewrite replaced. A malformed `LoadData` descriptor stalling MTE1 forever
is exactly the symptom.

Replace the hand-rolled descriptors in `Gemm128`, `Gemm16`, `LoadBt` (kernel1)
and `Gemm`, `GemmAt`, `LoadNd2Nz` (kernel2) with the catlass tile classes, which
take layout objects and derive every stride themselves:

- `Catlass::Gemm::Tile::CopyGmToL1<ArchTag, GemmType<Element, Layout>>`
- `Catlass::Gemm::Tile::CopyL1ToL0A<...>` / `CopyL1ToL0B<...>`
- `Catlass::Gemm::Tile::CopyL0CToGm<...>`

Build layouts with `layout::RowMajor(rows, cols)` and
`layout::zN::MakeLayout<Element>(rows, cols)`. Model the structure on
`examples/19_mla/mla_kernel.cpp`. This also deletes `utils.hpp`'s
`ZnBlockOffsetBytes` / `Nd2NzC0Stride` / `Nd2NzNStride`.

Bisect within `RunLMqk` first — it is the simpler of the two cube phases (one
`LoadBt` plus two `Gemm128` calls), and whatever fixes it very likely fixes
`RunNeumann` too.

Kernel2 still uses cross-core handshakes and must get the same split treatment
once kernel1's cube path runs.

## Diagnostics left in the tree

`src/fwd_kernel1.asc` carries three probe kernels, none on the real path:

- `FlashKdaNoop` / `_C.noop(tensor)` — smallest kernel on this launch path.
- `FlashKdaAivOnly` / `_C.aiv_only(src, dst)` — AIV vector work, no sync. This
  is the one that produced exact `exp` output.
- `FlashKdaSyncOnly` (`FLASH_KDA_SYNC_ONLY=1`) and `FlashKdaSyncSmall`
  (`FLASH_KDA_SYNC_SMALL=1`) — handshake only, large and small arguments.

`tests/sync_probe.cpp` is the standalone comparison point; build it with the
command in its header comment.

To re-run the phase bisection, stub any subset of `RunPrepare` / `RunLMqk` /
`RunMask` / `RunNeumann` with an early `return;`.

## Two places the code lives

Local `~/Documents/flash-kda-yaoyaolingxian` is the git repo (3 commits pushed).
**macOS has since blocked `~/Documents`** for this shell — every access returns
`Operation not permitted`, including with the sandbox disabled. That is TCC
(System Settings → Privacy & Security → Files and Folders / Full Disk Access),
not something I can grant myself. So the last round of work is *not* committed.

The uncommitted-but-working sources are in two safe places:

1. `scratchpad/fromcard/` in this repo — extracted copies of every file
   (`include/flash_kda/*.hpp`, `src/*.asc`, `src/flash_kda.cpp`, tests).
2. `/user/zhouyiran/flashkda/FlashKDA-Ascend` on Cybertron — JuiceFS, shared
   and persistent, so it **survives the pod expiring**. This is the copy that
   actually builds; treat it as authoritative.

To finish the commit once `~/Documents` is reachable again: copy
`scratchpad/fromcard/{include,src}` over the repo's `FlashKDA-Ascend/` and
commit. Or re-pull from the card.

## The card

Devspace job **645140**, 1× 910B3 preemptable, wulan, CANN 8.5.0. Expires
**22:48 today** and cannot be extended (`/api/job/lifecycle` only accepts
notebook jobs). Re-create with the recipe in `docs/bringup.md` — the workspace
under `/user/zhouyiran` persists, so a new pod picks up where this left off.

Rebuild on a fresh pod:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/usr/local/python3.11.14/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_LAUNCH_BLOCKING=1
cd /user/zhouyiran/flashkda/FlashKDA-Ascend/build
make -j8
cp _C.cpython-311-aarch64-linux-gnu.so ../flash_kda/
cd .. && PYTHONPATH=$PWD python3 test_npu_nocompute.py
```

## What this image cannot do

The image ships the AscendC compiler and headers but **not the precompiled
built-in operator library**:
`opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/` contains only `ops_oam` —
no `Cast`, no `Add`, none of the usual ~2000 operator binaries.

Consequences:

- Every `torch_npu` **compute** op fails with `AclOpKernelInit failed` /
  error 561103. Even `torch.randn(device="npu")` fails.
- Allocation, H2D and D2H copies **do** work (they are memcpy, not kernels).
- Custom AscendC kernels work fine — catlass's `00_basic_matmul` reports
  `Compare success` on this card.

So `tests/test_fwd.py` cannot run as written. `test_npu_nocompute.py` was added
for this: it generates inputs and computes the reference **on CPU**, and uses
the device only to hold inputs and return results. Keep using that until an
image with the full opp package is available.

Two edits were needed to make the reference CPU-runnable:
- `tests/torch_ref.py` used `torch.mm(..., out_dtype=torch.float32)`, which is
  NPU-only. With fp32 inputs the argument is a no-op, so it became
  `torch.mm(a.float(), b.float())`.
- `tests/test_fwd.py`, `tests/torch_ref.py` and `flash_kda/__init__.py` never
  imported `torch_npu`, so the `npu` device was unregistered.

## The hang — what is established

**The kernel hangs on device.** Confirmed cleanly:

```
launching
launch returned (async)      <- fwd() returns, the launch is asynchronous
<torch_npu.npu.synchronize() never returns>
```

Two false leads worth not repeating:

- `COMPLETED` printing after `fwd()` does **not** mean success. The launch is
  async, so the print happens before the kernel runs. Always
  `torch_npu.npu.synchronize()` before believing anything.
- The process appearing to "hang at teardown" was the same thing: exit was
  synchronizing a kernel that never finished. A control that allocates on NPU
  and copies without calling `fwd` exits cleanly (`exit=0`), so the device and
  runtime are fine — the fault is in my kernel.

It reproduces on the **smallest possible input** — `B=1, T=16, H=1`, a single
tile on a single core — and it reproduces every run. That is good news for
debugging: one tile, one AIC, one AIV pair, fully deterministic.

### Ruled out — and this narrows it a lot

Bisection result, which redirects everything: **kernel1 hangs with all compute
removed.** With `FKDA_SKIP_AIV` and `FKDA_SKIP_AIC` both defined (the stage
gates are in the tree) so each side executes nothing but its handshakes, *and*
`launch_fwd_recurrence` commented out so only kernel1 launches, a single tile
still never synchronizes.

So the fault is **not** in the algorithm, the GEMM parameters, the fractal
strides, or the Neumann iteration. It is in the launch/synchronization
scaffolding. Everything below follows from that:

- Not the algorithm — there is none left running.
- Not kernel2 — it is not being launched.
- Not the B-operand `srcStride` — reverted, no effect either way.
- Not both AIV subcores signalling — guarding on `subIdx == 0` changed nothing
  (the guard is still in the tree; it is defensible but not the bug).

### The one difference from the probe that works

`tests/sync_probe.cpp` runs the *same* two-round handshake and completes. The
differences between it and the real kernel are now the entire suspect list:

| | sync_probe (works) | kernel1 (hangs) |
|---|---|---|
| blockDim | 1, 2 and 4 all complete | 1 (`total_tiles * H`) |
| stream | own `aclrtCreateStream` | torch's `c10_npu::getCurrentNPUStream()` |
| launched from | plain C++ `main` | pybind module inside torch |

### The bisection, as far as it got

Every row below was run on the 910B3. The handshake is identical in all of them
— two rounds, `CrossCoreSetFlag<0x2, PIPE_MTE3>` from AIV against
`CrossCoreWaitFlag` on AIC and back.

| Variant | Where it runs | Result |
|---|---|---|
| `sync_probe`, blockDim 1 / 2 / 4 | standalone binary | **completes** |
| `sync_probe` + `Resource<ArchTag>` member | standalone binary | **completes** |
| `sync_probe` + `PipeBarrier<PIPE_ALL>` | standalone binary | **completes** |
| `noop` kernel, no handshake | inside the .so, torch stream | **completes** (writes 42) |
| `sync_only`: handshake only, `FwdParams` by value | inside the .so | **hangs** |
| `sync_small`: handshake only, two scalar args | inside the .so | **hangs** |
| `FwdPrepareKernel`, all compute compiled out | inside the .so | **hangs** |

So: the handshake works in a standalone binary and hangs inside the Python
extension, and a kernel *without* a handshake works fine inside that same
extension. The two ingredients are individually fine and fail in combination.

Also ruled out along the way, each by experiment rather than reasoning:

- blockDim (1 works standalone).
- `Catlass::Arch::Resource` / its `TPipe` — added to the probe, still completes.
- `PipeBarrier<PIPE_ALL>` — added to the probe, still completes.
- torch's stream — launching on a private `aclrtCreateStream` with an explicit
  synchronize still hangs.
- Kernel argument size — `sync_small` takes two scalars and still hangs.
- `aclrtGetHardwareSyncAddr` failing — it returns `rc=0` and a plausible
  address (`0x3fffffb9000`) inside the extension.
- Missing link libraries — the module now links exactly what catlass's examples
  link (`-L$CANN/aarch64-linux/devlib`, plus `ascendc_runtime`, `profapi`,
  `mmpa`, `ascend_dump`, `c_sec`, `error_manager`, `nnopbase`, `ascendalog`,
  `unified_dlog`, `ascend_hal`) and compiles with catlass's two extra `-mllvm`
  codegen options. Still hangs. **Keep this change** — the old link line was
  genuinely incomplete even if it is not the bug.
- A missing mix-kernel task-type declaration — no `KERNEL_TASK_TYPE`,
  `KERNEL_TYPE_MIX_AIC_1_1` or `SetKernelTaskType` exists anywhere in catlass or
  in this CANN 8.5.0's `tikcpp` headers, so there is no such knob to set.

### Where to pick this up

The evidence says the problem is how a cross-core-synchronized kernel is
*launched from a shared library inside the torch process*, not anything about
the kernel body. Two concrete next steps:

1. **Reproduce it standalone-but-shared.** Build `sync_probe` as a `.so` with a
   tiny C `main` that `dlopen`s it, no torch at all. If it hangs, the variable is
   "kernel in a shared object" and the answer is probably a device-binary
   registration issue — the `.o` for a `.asc` file inside a shared module may not
   get its FFTS/sync section wired up the way a linked executable does. Compare
   `readelf -S` on the standalone binary against the `.so`.
2. **Ask the other direction.** Does any *documented* CANN example ship an
   AIC/AIV cross-core kernel inside a Python extension? catlass's
   `.agents/skills/catlass-example-to-torch-intf` template exists precisely for
   torch integration — check whether every example it supports is single-core-type
   (`00_basic_matmul` is pure AIC). If so, cross-core sync inside a torch
   extension may be unsupported, and kernel1 should be restructured to avoid it:
   split it into two separate kernel launches (an AIV-only prepare and an
   AIC-only GEMM pass) communicating through GM, with the stream providing the
   ordering instead of CrossCore flags.

Option 2 is the more likely resolution and is also a simpler design. It costs one
extra kernel launch per tile-group and removes cross-core sync from the project
entirely.

### Diagnostics left in the tree

`src/fwd_kernel1.asc` carries three extra kernels, all reachable without
touching the real path:

- `FlashKdaNoop` / `_C.noop(tensor)` — smallest kernel on this launch path.
- `FlashKdaSyncOnly` — handshake only, `FwdParams` by value.
  Set `FLASH_KDA_SYNC_ONLY=1`.
- `FlashKdaSyncSmall` — handshake only, scalar args.
  Set `FLASH_KDA_SYNC_SMALL=1`.

`FKDA_SKIP_AIV` / `FKDA_SKIP_AIC` / `FKDA_SKIP_NEUMANN` compile out kernel1's
stages while keeping handshake counts symmetric. `tests/sync_probe.cpp` is the
standalone comparison point; build it with the command in its header comment.

### Suspects for later, once it runs at all

1. **`Sigmoid`'s `Div` and `ReduceSum`'s work buffer** — both are higher-level
   vector APIs that may want a reserved tmp buffer, and `Resource<ArchTag>`
   hands all 192 KB of UB to `ubBuf`, leaving none.
2. **`ComputeNeumann`'s Fixpipe-to-bf16 chain** — nine dependent
   MMAD→Fixpipe→GM→L1 round trips with `FIX_MTE2` as the only guard between a
   Fixpipe and the next load of the slot it just wrote.
3. **`GemmAt`'s `ifTranspose = true`** on the nZ A operand in kernel2.
4. The hand-derived stride refactor described above — still worth doing on its
   own merits, but it is now clear it is not what is hanging.

`msDebug` is available in this CANN install if needed.

### A wrong turn, recorded so it is not repeated

I "fixed" `Gemm128`'s B operand from `srcStride = 1` to `8`, reasoning from
catlass's `layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL` with B as `[128,16]`. It
is reverted: it did not help, and I could not verify the L1 layout it assumes.

The real lesson is that I was hand-deriving fractal strides at all — the exact
habit that produced the draft this rewrite replaced. Every one of these numbers
is something catlass computes from a layout object.

**Do this before more bisection:** replace the hand-rolled `Nd2NzParams`,
`LoadData2DParams` and `FixpipeParamsV220` in both kernels with the catlass tile
classes, which take layouts and derive the strides themselves:

- `Catlass::Gemm::Tile::CopyGmToL1<ArchTag, GemmType<Element, Layout>>`
- `Catlass::Gemm::Tile::CopyL1ToL0A<...>` / `CopyL1ToL0B<...>`
- `Catlass::Gemm::Tile::CopyL0CToGm<...>`

Build layouts with `layout::RowMajor(rows, cols)` and
`layout::zN::MakeLayout<Element>(rows, cols)`; model the structure on
`examples/19_mla/mla_kernel.cpp`. That also deletes `utils.hpp`'s
`ZnBlockOffsetBytes` / `Nd2NzC0Stride` / `Nd2NzNStride`, which exist only
because the code does this arithmetic itself.

There is a decent chance the hang disappears with that refactor, since a
malformed `LoadData` descriptor is exactly the kind of thing that stalls MTE1
forever.

### If it survives the refactor

Bisect kernel1 by returning early from `operator()<AIV>` / `operator()<AIC>`
after each stage, keeping the handshake counts symmetric so the other side never
blocks. Remaining suspects:

1. **`Sigmoid`'s `Div` and `ReduceSum`'s work buffer.** Both are higher-level
   vector APIs that may want a reserved tmp buffer; `Resource<ArchTag>` hands
   all 192 KB of UB to `ubBuf`, leaving none.
2. **`ComputeNeumann`'s Fixpipe-to-bf16 chain** — nine dependent
   MMAD→Fixpipe→GM→L1 round trips with `FIX_MTE2` as the only guard between a
   Fixpipe and the next load of the slot it just wrote.
3. **`GemmAt`'s `ifTranspose = true`** on the nZ A operand in kernel2.

`msDebug` is available in this CANN install if bisection stalls.

## Correctness work still outstanding

Nothing has been numerically validated yet — the hang blocks it. Once it runs,
`test_npu_nocompute.py` prints `err_ratio` against the CPU reference; bf16 end
to end should land around 1e-2. Then extend to the cases the rewrite claims to
support but that have never executed: `initial_state` / `final_state` in both
bf16 and fp32, `cu_seqlens` varlen with a short tail chunk, and `H > 1` with
`T` not a multiple of 16.

The algorithm fixes listed in commit `d3a004c` (the log2(e) gate_scale, the
double `exp(g_total)`, the aliased `inv_cumsum`, the swapped Nd2Nz strides) are
all reasoned from the CUDA reference and the catlass sources, **not** confirmed
by a passing test. Treat them as high-confidence but unverified.
