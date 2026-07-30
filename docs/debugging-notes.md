# FlashKDA-Ascend — session handoff, 2026-07-30 evening

## Where things stand

Both kernels are **rewritten and compile as real device objects**, the Python
extension **builds and links**, and the kernel **launches on a real 910B3**.
It currently **hangs in the compute** with `aicore execution times out`.

That is the honest state: from "never compiled" to "compiles, links, launches,
hangs". The remaining bug is a live debugging problem, not a missing rewrite.

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

### Ruled out

- **The cross-core handshake pattern.** `scratchpad/sync_probe.cpp` runs
  kernel1's exact two-round AIC/AIV handshake with all compute removed and
  completes (`sync result: 0`). Caveat: its 8 `uint32` markers per core share
  one 32-byte cacheline so AIC/AIV clobber each other's — give them separate
  cachelines before trusting per-stage detail.
- **Both AIV subcores signalling.** I guarded the handshakes on `subIdx == 0`
  on the theory that double-setting `elemReady` left unconsumed counts. It did
  not change the hang. The guard is still in the tree and is probably correct
  on its own merits, but it is not the bug.
- **The B-operand `srcStride`.** See below.

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
