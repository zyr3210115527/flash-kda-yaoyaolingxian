# FlashKDA-Ascend — session handoff, 2026-07-30 evening

## Where things stand

**Kernel1 works and is verified on hardware.** All four phases run clean and
every output is checked against float64 CPU expectations on a 910B3:

```
k_decayed  1.478e-03    q_decayed  1.674e-03
k_inv      1.785e-03    k_restored 1.734e-03
g_total    8.962e-07    Mqk        3.851e-03
INV        1.047e-02
```

It holds at T=64, H=4 (16 cores), not just the single tile it was debugged on.

**The full forward pass runs end to end.** Both kernels execute, output is
finite and non-zero.

**Kernel2 is not correct yet.** `err_ratio ~ 0.44` against torch_ref, where the
bf16 floor is ~1e-2. And it is **non-deterministic**: repeated runs of the same
binary on the same input give 4.3917e-01, 4.3926e-01, 4.3930e-01, and roughly
one run in three or four produces no number at all. A deterministic kernel on
fixed inputs must be bit-identical, so there is still a race.

Getting from 1.0 to 0.44 came from correcting the recurrence; the remaining gap
and the non-determinism are unresolved.

## What to chase next, in order

1. **The race — now localized to the chunk-to-chunk state carry.** Capping the
   chunk loop at one makes kernel2 **bit-deterministic**: four runs give
   err_ratio=8.8452e-01 exactly, no faults. Uncapped, the same binary gives
   4.3917e-01 / 4.3926e-01 / 4.3930e-01 and fails outright about one run in
   three. So nothing inside a chunk races; the fault is in what one chunk hands
   the next, which is only the recurrent state in GM at `StateOff()`.

   `DecayState` (AIV) writes it at the end of chunk t; `StateToBf16` (AIV) reads
   it at the start of chunk t+1. The writer-side flush already added is not
   enough — `DataCacheCleanAndInvalid` pushes the line out of the *writer's*
   cache, but the reader is a separate launch and may still hold a stale copy of
   that address. **Invalidate on the read side before loading the state** is the
   next thing to try.

   (The single-chunk 0.8845 is not comparable to the uncapped 0.4393 — capping
   leaves later chunks unprocessed, so most of the output is missing. It is a
   determinism probe only.)

   Three things were fixed along the way and none of them was the race:
   - `DataCacheCleanAndInvalid` on the six workspace stores that cross a phase
     boundary. This *did* remove the intermittent aicore exceptions.
   - The workspace was `torch.empty`; it is now zeroed.
   - `DecayState` read `g_total` with `GetValue` (scalar unit) behind only an
     `MTE2_V` barrier; it now also issues `MTE2_S`.

   Still to examine: the AIV stores in `FinishOut` and `FinishChunk` against
   their own MTE3 completion, and whether `Drain`'s Fixpipe targets need
   flushing the way the AIV stores did. A useful experiment is to run with
   `maxChunks` capped at 1 — if a single chunk is deterministic, the race is in
   the chunk-to-chunk carry, which narrows it to the state in `DecayState`.

2. **The B-operand layout.** `Gemm` and `GemmAt` load B as RowMajor→zN and issue
   `ifTranspose = false`. L0B accepts only zZ (transposing) or nZ (not), so this
   is a well-formed but wrong operand — plausibly most of the residual 0.44. The
   full zZ parameterization is derived below. I implemented it once and it hung;
   that was before the argument-shape fix, so it deserves a retry now that the
   launch path is sound.

3. **Then re-verify.** `tests/check_prepare.py` for kernel1, then
   `tests/test_npu_nocompute.py` for the end-to-end number, then
   `tests/validate_ref.py` stays the oracle.

## The next thing to fix

Three separate problems in kernel2. The first was found by reading and needs no
hardware.

### 1. The recurrence is the wrong formula, and beta is missing entirely

`beta` appears **zero times** in all of `fwd_kernel2.hpp`. The delta-rule gate
is not implemented at all. And the `u` being computed is not the right quantity:

```
kernel2:    u = Mqk @ v  -  (INV @ k_dec) @ state
reference:  u = INV @ ((v - k_dec @ state) * sigmoid(beta))
```

The first term uses `Mqk` where it should use `INV`, and the beta scaling is
absent. `(INV @ k_dec) @ state` is fine in isolation -- associativity makes it
equal to `INV @ (k_dec @ state)` -- but it cannot be combined with the other
term as written. On its own this explains `err_ratio ~ 1`.

The correct decomposition keeps the same five-phase, five-launch shape:

```
1  AIV   load v and beta; state -> bf16
2  AIC   t1    = k_dec @ state ;  out_acc = q_dec @ state
3  AIV   v_sub = (v - t1) * sigmoid(beta)
4  AIC   u     = INV @ v_sub ;  s8 = Mqk @ u ;  s7 = k_res^T @ u
5  AIV   out   = out_acc + s8 ;  state = state * g_total + s7
```

This is what the inherited draft's own comments described; the implementation
drifted from them.

### 2. The B operand goes through a layout L0B does not accept

L0B accepts only `zZ -> nZ` with `ifTranspose = true`, or `nZ -> nZ` with
`ifTranspose = false`. kernel2 loads B from GM as RowMajor->zN and issues
`ifTranspose = false` -- neither. That is the shape of bug that yields a
well-formed but wrong operand: no fault, uncorrelated output.

Kernel1 avoids it because `LoadBt` writes nZ directly via the ColumnMajor
parameterization. Kernel2's B operands (v, u, state) are genuinely RowMajor
`[k, n]`, and a ColumnMajor view gives B-transpose, not B -- so they must go
through zZ. Derived from `CopyGmToL1<RowMajor, zZ, B1>`, `zZ::MakeLayout` and
`CopyL1ToL0B<zZ, nZ>`:

    GM RowMajor [k, n] -> L1 zZ
      ndNum             = k / 16
      nValue            = 16
      dValue            = n
      srcNdMatrixStride = 16 * n
      srcDValue         = n
      dstNzC0Stride     = zZ.stride(3) / C0 = 256 / 16 = 16
      dstNzNStride      = zZ.stride(0) / C0 = 16 / 16  = 1
      dstNzMatrixStride = zZ.stride(1)      = roundUp16(n) * 16

    L1 zZ -> L0B nZ
      repeatTimes = ceil(n / 16), srcStride = 1, dstGap = 0, ifTranspose = true
      looped over ceil(k / 16) fractal rows, stride roundUp16(n) * 16 both sides

I implemented exactly this and it hung, so something in it is still off. The
uint16 stride fields and the exact-fit L1 slots are what I would check first;
`scratchpad/patch_k2_l1slots.py` widens those slots to 64 KB (untested), and the
attempt itself is `scratchpad/patch_k2_bpath.py`.

### 3. Kernel2 hangs regardless of its phase bodies

The phase bisection after the buffer change gives HANG for every variant
**including v0, where all seven phases return immediately**. So the hang is not
in any body -- it is the launch structure. Kernel1 issues 4 launches per call
and works; kernel2 issues `1 + 5*chunks + 1`, which is 22 at T=64.

`tests/probe_launchcount.sh` caps the chunk loop at 0/1/2/4 to find the
threshold. Two candidates: launch count itself, or argument shape -- kernel2's
entries take `(FwdParams, int32_t)` while kernel1's take a single `FwdParams`.
`scratchpad/patch_k2_chunkparam.py` moves the chunk index into `FwdParams` so
the signatures match, removing that variable (untested).

### A fallback if the launch count is the cause

If `probe_launchcount.sh` shows the hang scales with launches, the five-per-chunk
structure is not viable and there is no way to shrink it: the recurrence is
sequential in chunks and each chunk genuinely alternates AIV/AIC four times.

The escape is to stop alternating. Run the whole of kernel2 as a **single
AIV-only kernel** with the chunk loop inside it, doing the GEMMs on the vector
unit. For D = 128 a `[16,128] @ [128,128]` product is 262144 MACs, which the
vector unit can do correctly, just slowly. That is one launch instead of
5*chunks, needs no cross-core sync, and no cube layouts at all -- so it sidesteps
the B-operand problem too.

It would be markedly slower than the cube, which matters eventually. But it
would give a correct forward pass to measure against, and the cube path can be
reintroduced afterwards with a known-good reference to diff against. Getting
something numerically right first is worth more here than keeping the fast
structure that has never produced a correct number.

The durable alternative to all this hand-derived layout work is to call the
catlass tile classes directly. Every layout bug in this project so far has been
a hand-derived stride.

## What was actually wrong (all confirmed on hardware)

Five distinct bugs, in the order they were found:

1. **Cross-core sync does not work from a Python extension.** The identical
   handshake completes in a standalone binary at any blockDim, with a Resource
   member, with PipeBarrier — and hangs inside the .so with any argument shape.
   A kernel without a handshake works fine in that same .so. Ruled out along the
   way: blockDim, `Catlass::Arch::Resource`, torch's stream, argument size,
   `aclrtGetHardwareSyncAddr`, and missing link libraries.
   Fix: split each kernel into single-core-type launches ordered by the stream.

2. **Missing core-type guards.** Splitting dropped the `operator()<AIC>` /
   `<AIV>` specialization that was implicitly guarding each body. The compiler
   builds every kernel as a mix binary with `_mix_aic` and `_mix_aiv` halves, so
   the cube phases ran their L1/L0 code on a vector core — "The MPU address
   access is invalid".
   Fix: `if constexpr (g_coreType != AscendC::AIC/AIV) return;` in every phase.

3. **`LoadBt`'s Nd2Nz parameters.** k_inv is `[CHUNK, D]` RowMajor and its
   transpose is the same bytes as ColumnMajor `[D, CHUNK]`. Nd2Nz describes a
   ColumnMajor source by its *columns*: `nValue` is the column count and
   `dValue` the column length. Having them swapped asked for 128 spans of 16 at
   a 128 pitch, reading to element 16272 of a 2048-element buffer.

4. **`Gemm16`'s `FIX_MTE2` event was unbalanced.** ComputeNeumann chains nine
   GEMMs, each reloading L1 from the GM the previous Fixpipe wrote. The set had
   no matching wait, so nine unconsumed events accumulated and the core stalled.
   Fix: a plain set/wait barrier at the end of each GEMM.

5. **The Neumann iteration squared (I − L) instead of L.** `(I-L)^2` is
   `I - 2L + L^2`, and the AIV never wrote plain L anywhere. The running product
   picked up an extra doubling per iteration — the 2³ = 8 that appeared on
   kINV's diagonal. Fix: the AIV fills L directly in the mask loop and stores it
   to scratch slot 5.

Two of my own testing bugs are worth recording because they cost real time:

- `struct` has no bfloat16 code, and `'e'` is IEEE fp16. Reading the four bf16
  workspace fields with `'e'` reinterprets the exponent and made correct data
  look wrong by factors of 20–500, while fp32 `g_total` passed. bf16 is the top
  16 bits of the fp32 pattern.
- The sync script built into `build/` but never installed the `.so` into
  `flash_kda/`, so two rounds of "identical results" were a stale binary.

## Next

1. **Kernel2's cube phases.** Same class of bug as kernel1: hand-rolled
   `Nd2NzParams` / `LoadData2DParams`. Start with `GemmAt`, whose `k_res^T`
   operand needs exactly the ColumnMajor parameterization `LoadBt` turned out to
   need. Bisect with the phase stubs, as for kernel1.
2. **Then end-to-end.** `tests/test_npu_nocompute.py` prints `err_ratio` against
   torch_ref; bf16 should land near 1e-2.
3. **Then the weak spots in the current verification.** With the test input
   |L| ≈ 0.028, so `(I+L)^-1 ≈ I` and kINV passing does not strongly exercise
   the Neumann series — a larger-L input would. And `L` itself differs from
   float64 by 6e-1 because `k_decayed @ k_inv^T` cancels ~28 orders of
   magnitude; torch_ref, which does the same bf16 arithmetic, is the fair
   yardstick there, not float64.
4. Replacing the hand-rolled descriptors with the catlass tile classes remains
   the durable cleanup, and would have prevented bugs 3 and (in kernel2)
   whatever is faulting now.

Beware: `fault kernel_name=` in plog can be stale, since the log accumulates
across runs. Confirm with `FLASH_KDA_SKIP_K2=1` before believing it.


## Blocked on you: Teleport login

The certificate expired at 04:08 on 2026-07-31, so `tsh ssh` and the Cybertron
job API both return 401. Re-auth is SSO:

```bash
"$HOME/Library/Application Support/Cursor/User/globalStorage/yangsuiyun.cybertron/bin/tsh" login \
  --proxy=teleport.cybertron.modelbest.co:443
```

Then create a devspace with the recipe in `docs/bringup.md`. The work on the
card is under `/user/zhouyiran/flashkda` (JuiceFS) and survived the pod.

Note also that macOS TCC intermittently blocks `~/Documents` for the shell, which
breaks git there. Everything is pushed to GitHub, so a `git pull` in the local
checkout is enough to catch up. If it recurs: System Settings → Privacy &
Security → Files and Folders.


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

**2. With the split in place, the hang is isolated to one call.** Running one
phase at a time, then bisecting inside the failing one:

| What runs | Core | Result |
|---|---|---|
| `RunPrepare` — normalize, gate, cumsum, decay, store | AIV | **completes** |
| `RunMask` — tril mask, beta sigmoid, build `(I - L)` | AIV | **completes** |
| `RunLMqk` — the L / Mqk GEMMs | AIC | **hangs** |
| `RunNeumann` — the inverse | AIC | **hangs** |
| `RunLMqk` with both `Gemm128` calls removed, `LoadBt` only | AIC | **hangs** |
| same, but `LoadBt` doing a plain non-transposed `[16,128]` load | AIC | **hangs** |
| same, but `Resource` constructed locally rather than as a member | AIC | **hangs** |
| `aic_only` probe — AIC writes one scalar to GM, no L1 | AIC | **completes** |

So the cube itself runs fine from the extension (`aic_only` returns 7), both
vector phases run correctly, and the failure is **one call**: the `Nd2Nz`
`DataCopy` from GM into L1. It hangs even in its simplest form — a plain
non-transposed `[16, 128]` load — so this is not about the transposed
parameterization.

Not the cause, each tested: `Resource` as member vs local; the transposed
ColumnMajor view; the two `Gemm128` calls (removed and it still hangs).

## What to do next

**First, get a card back.** The Teleport certificate expired at 04:08 on
2026-07-31, so both `tsh ssh` and the Cybertron job API return 401. Re-auth is
SSO and has to be done by you:

```bash
"$HOME/Library/Application Support/Cursor/User/globalStorage/yangsuiyun.cybertron/bin/tsh" login \
  --proxy=teleport.cybertron.modelbest.co:443
```

Then create a devspace with the recipe in `docs/bringup.md`. The work on the
card lives under `/user/zhouyiran/flashkda` (JuiceFS), so it survived the pod.

**Then run the experiment that decides the current hypothesis.**
`FlashKDA-Ascend/tests/aic_resource_probe.cpp` is a standalone three-stage
program, about two minutes end to end:

| Stage | What the AIC does |
|---|---|
| A | constructs `Catlass::Arch::Resource<ArchTag>`, returns |
| B | constructs its own `TPipe` + `TBuf<A1>`, returns |
| C | stage B plus one `Nd2Nz` GM→L1 `DataCopy` |

If **A hangs and B completes**, the hypothesis holds: `Resource` constructs
*every* buffer unconditionally — including a 192 KB UB allocation through
`GetTPipePtr()->InitBuffer` — and the cube has no UB. The vector phases survive
it because asking a vector core for L1 is harmless; asking the cube for UB is
not. The fix for that is already committed (see below).

If **A completes and C hangs**, `Resource` is innocent and the fault is the
`DataCopy` itself; the next suspect is the L1 destination extent rather than the
descriptor, because the descriptor fields were verified correct — `CopyGmToL1`
derives `dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0`, and for a
`[16,128]` bf16 zN tile `zN::MakeLayout` gives `stride(3) = rowsRound * 16 = 256`,
so `dstNzC0Stride = 16` and `dstNzNStride = 1`, which is exactly what the
hanging code passes.

### The candidate fix, already committed but untested

Kernel1 no longer uses `Catlass::Arch::Resource`. Each phase constructs a holder
with only the buffers its core type owns, sized to the phase rather than to the
whole on-chip capacity:

- `K1AivBufs` — one `TPipe` + `TBuf<VECCALC>` of `K1Ub::kEnd` bytes.
- `K1AicBufs` — one `TPipe` + `TBuf<A1>` of `K1L1::kEnd`, plus L0A/L0B of 4 KB
  each and L0C of 1 KB.

This is the same shape as the `aiv_only` probe that returned numerically exact
results on the card, which is the only buffer pattern proven to work in this
environment.

**It has not been compiled or run** — the card expired before it could be. Treat
it as a hypothesis with code attached, and be ready to revert if stage A of the
probe completes.

### After that

Once a card is back, in this order:

1. **`tests/check_prepare.py`** — verifies kernel1's prepare output
   (k_decayed, q_decayed, k_inv, k_restored, g_total) field by field against
   float64 CPU expectations. Those AIV phases already run, so this should pass
   immediately and confirms the gate, cumsum and decay math on hardware. Its
   `--cpu-only` mode already passes.
2. **`tests/aic_resource_probe.cpp`** — decides the Resource hypothesis.
3. Whichever fix that implies, then `tests/test_npu_nocompute.py` for the first
   end-to-end number against the reference.

The durable cleanup remains replacing the hand-rolled `Nd2NzParams` /
`LoadData2DParams` / `FixpipeParamsV220` in `Gemm128`, `Gemm16`, `LoadBt`
(kernel1) and `Gemm`, `GemmAt`, `LoadNd2Nz` (kernel2) with the catlass tile
classes, which take layouts and derive every stride themselves:

- `Catlass::Gemm::Tile::CopyGmToL1<ArchTag, GemmType<Element, Layout>>`
- `Catlass::Gemm::Tile::CopyL1ToL0A<...>` / `CopyL1ToL0B<...>`
- `Catlass::Gemm::Tile::CopyL0CToGm<...>`

Build layouts with `layout::RowMajor(rows, cols)` and
`layout::zN::MakeLayout<Element>(rows, cols)`; model on
`examples/19_mla/mla_kernel.cpp`. This also deletes `utils.hpp`'s
`ZnBlockOffsetBytes` / `Nd2NzC0Stride` / `Nd2NzNStride`.

**Kernel2 still uses cross-core handshakes and therefore cannot run at all.** It
needs the same split kernel1 received. It is serial across chunks with two
handshakes per chunk, so the chunk loop has to move to the host — roughly five
launches per chunk. This was deliberately not attempted yet: kernel2 consumes
kernel1's workspace, so it cannot be debugged until kernel1's cube path works,
and stacking another large untested change would make it hard to tell which
change broke what.

Its state transpose at the GM boundary **is** done, since
`validate_ref.py` check [4] specified it precisely.

## Untested changes currently in the tree

Two, both committed with reasoning but never compiled — the card expired first:

- `K1AivBufs` / `K1AicBufs` replacing `Catlass::Arch::Resource` in kernel1.
- The `StateGmToUbT` / `StateUbToGmT` boundary transpose in kernel2.

If the first turns out wrong (stage A of the probe completes), revert it before
debugging anything else.

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
