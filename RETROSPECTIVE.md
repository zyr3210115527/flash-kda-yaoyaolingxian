# Five days on a Huawei NPU

*A retrospective on porting FlashKDA from CUDA to Ascend C: what happened, what
I got wrong, and what the ecosystem is actually like to work in.*

---

## The shape of it

I was handed an unfinished port. Roughly 2000 lines of Ascend C existed, and
none of it had ever run — no kernel entry point, empty launch functions, a build
system that did not link the Ascend runtime. The instruction was to rewrite both
kernels, get them onto a card, and make them fast.

Eighty-six commits later:

| | |
|---|---|
| Forward pass | correct, 12/12, bit-identical across runs |
| At the CUDA reference's own shape | 32.9 ms vs 1.62 ms on an H20 — **20×** |
| First measurement | 790× |
| First measurement, taken correctly | ~94× |

That third row is the most honest thing in the table, and I will come back to
it.

---

## Act I: getting anything to run at all (day 1)

The first day produced no working code and was not wasted. It went:

```
25f6880 Build and launch on hardware; kernel still hangs on device
0cac78c Bisect the hang down to the sync scaffolding, not the algorithm
945f8f7 Narrow the hang to cross-core sync inside the Python extension
dec5642 Isolate the hang to a single Nd2Nz GM-to-L1 DataCopy
```

Four commits, each one narrowing a hang. That is what bring-up on unfamiliar
silicon looks like: you cannot debug the algorithm until you can trust the
machinery, and the machinery gives you almost nothing to go on. A hang is a
hang. No stack, no assertion, no timeout with a line number — the process just
stops and you kill it.

The `Nd2Nz` bug was my introduction to the fractal layout. Ascend's cube unit
does not eat row-major matrices; it eats 16×16 fractal tiles in layouts called
zN, zZ and nZ, and you describe the transformation with a struct of six strides.
I had described a ColumnMajor source by its *rows*. The correct reading is that
`nValue` is the **column** count and `dValue` is the length of a column. Get it
backwards and you ask the hardware to read element 16272 of a 2048-element
buffer, which it does, without complaint, until something downstream is NaN.

Every layout bug in this project was a hand-derived stride. All of them.

---

## Act II: correctness, and the pleasure of a good oracle (days 1–2)

The single best decision was building `torch_ref.py` — a CPU emulator that
reproduces the CUDA kernel's arithmetic *bit for bit*, including its fp16
accumulation, its `sigmoid` polynomial approximation, and its rounding order.
Then verifying the oracle itself against an independently written float64
implementation before trusting it for anything.

This paid for itself immediately, because it turns "the output is wrong" into
"field `k_inv` diverges at row 3, column 40". Bugs found this way:

- **The Neumann iteration squared `(I−L)` instead of `L`.** `(I−L)² = I − 2L +
  L²`, and the symptom was a clean `2³ = 8` sitting on the inverse's diagonal.
  Wrong answers that are *tidy* are a gift.
- **`DataCopyPad` pads every block to 32 bytes.** Gathering one bf16 per token
  does not land them contiguously — they land 16 elements apart with garbage
  between. This corrupted `beta` in both kernels and every quantity downstream
  of it.
- **`struct` has no bf16 format code, and `'e'` is IEEE fp16.** This one was
  mine, in a test, and it made correct data look wrong by 20–500×. I spent
  hours debugging a kernel that was fine.

And the one I am least proud of: my sync script rebuilt the `.so` but did not
install it. Two rounds of "identical results, no change" were a stale binary.

**Lesson, unglamorous but load-bearing:** when a result is surprising, suspect
the measurement before the system. I did not, twice, and it cost more than any
kernel bug.

---

## Act III: the 790× that wasn't (day 3)

The port worked. I benchmarked it. It was 790× slower than CUDA.

That number was in the repo for exactly one commit:

```
99a2cf9 Benchmark the forward pass: ~790x the CUDA reference, and why
46a169c Correct the benchmark: 90% of it was the per-call workspace allocation
```

The Python wrapper allocated the workspace on every call — 302 MB of host-side
zeros, copied to the device, per invocation. It was 88–98% of everything I had
measured. The kernel was never 790× slow; my harness was.

The tell was available and I nearly missed it: I stubbed out *every compute
phase* of kernel1 and the total moved by 2%. If removing all the work changes
nothing, you are not measuring the work.

Fixing it properly — cache the buffer, allocate on device, stop zeroing it once
a NaN-poison test proved nothing reads before writing — took end-to-end time
from **29.96 ms to 2.73 ms**. An 11× speedup that was, in a sense, not an
optimisation at all. It was the removal of something that should never have
been there.

It also unlocked the honest comparison. The 19 GB workspace needed for the
CUDA reference's own benchmark shape could not go through the host; on device
it fit. Only then could I compare like with like.

---

## Act IV: measuring properly, and being wrong in public (days 3–4)

With the harness fixed, a pattern emerged that repeated for two days: **I would
reason about what should be slow, act on it, and be wrong.**

| Prediction | Result |
|---|---|
| Batching the L2 normalisation is the biggest win | No measurable change |
| Grid-stride will cut dispatch cost | No change (correct at the time!) |
| The scalar round trips in `DecayState` are the cost | No change, twice |
| Casting the state from UB saves 2 GB of reads | Worth nothing — it was L2-resident |
| Halving the handshakes recovers 2.2 ms | Worth exactly 0.00 ms |

That last one deserves its own confession. I computed "AIC busy 4.8 ms + AIV
busy 3.3 ms = 8.1 ms in a 10.3 ms kernel, so 2.2 ms is handshake latency" —
except `aic_*_ratio` is a fraction of `aicore_time` and `aiv_*_ratio` is a
fraction of `aiv_time`. **Different denominators.** Summing them is meaningless.
Done correctly the gap is 0.64 ms. I optimised away a quantity that did not
exist, and the restructure is still in the tree, marked as a null result,
because it is strictly fewer synchronisation points for the same work.

What actually worked, once I started measuring first:

- **Fusing kernel2 into a single kernel** — 2051 launches became 1. Worth
  47× → 38×, the single largest win.
- **Keeping the Neumann chain in L1** — six GM round trips became one. 38× → 30×.
- **The recurrent state resident in UB** across the whole chunk loop. 30× → 28×.
- **Using the second vector core** — 27× → 21×.
- **Shrinking the workspace** from 18.6 GB to 6.3 GB by sizing scratch slots to
  what they actually hold, rather than all to the widest.

There is a sharp lesson buried in the discards. The "cast the state from UB"
idea measured as *worthless* the first time and *paid* the second — because in
between, launches stopped dominating and `DecayState` became the largest phase.

> **A negative result is only valid against the configuration it was measured
> in.** I now write that in the commit message when I discard something.

---

## Act V: the two-sentence fix (day 5)

Twice I concluded that cross-core synchronisation between the cube and vector
units simply does not work from a Python extension. I designed the whole
four-launches-per-chunk structure around that belief. It was wrong.

The missing ingredient is `KERNEL_TASK_TYPE_DEFAULT(...)` in the kernel body.
Without a declared task type the runtime never provisions the FFTS sync
resources, so the first `CrossCoreWaitFlag` waits forever. With it, the same
probe that had hung for weeks completed instantly.

I found it sideways — looking for a way to stop mix-mode launches starting
blocks on both core types, I hit `KERNEL_TASK_TYPE`, and the two problems turned
out to be the same problem. No amount of varying the thing I *was* varying
(argument count) would ever have found it.

That unlocked the fusion, and the fusion was worth more than everything before
it combined.

Then Kimi (of Moonshot AI, following the repo through the GitHub API and
relaying to the repo owner) sent official documentation, and one line of it
unlocked the next win. Mode `0x2` is AIC↔AIV **within one AI Core** — "每个
AICore 内部，2 个 AIV 等 1 个 AIC". I had already found experimentally that
every AIV must set the flag; what I had missed is that the sync being
*per-core* constrains the **work assignment**, not just the flags. My split gave
AIV block `i` unit `i` globally, so the two vector cores on a core were working
on units `2j` and `2j+1` while their own cube core was on unit `j`. It produced
correct output for exactly the one-unit shape and NaN everywhere else — which I
had misdiagnosed as a layout bug and abandoned after six attempts.

The fix was one line of indexing. It was worth 31% of kernel1.

**Six failed attempts, and the thing that broke the deadlock was a sentence of
documentation from someone who had never touched the hardware.**

---

## What I got wrong, collected

Because the failures are the useful part:

1. **Trusted a single test run.** Twice. Both times an intermittent race hid
   behind one clean pass. `race_probe.py` now needs six runs to mean anything,
   and the suite is run repeatedly, not once.
2. **Summed ratios with different denominators**, and optimised the imaginary
   gap it produced.
3. **Reported a kernel corruption that was my own diagnostic** — I read a
   workspace slot without `FLASH_KDA_SKIP_K2`, so kernel2 had already reused it.
   The tell was that the "corrupt" values scaled exactly 2× with the inputs,
   which a structural identity cannot do. The test now verifies its own
   addressing before trusting anything it reads.
4. **Wrote a test that proved nothing and passed.** The first Neumann strength
   test drove ‖L‖ to 0.082, where `I − L` is within the bf16 floor of the true
   inverse. It now refuses to pass when no case is discriminating.
5. **Believed a hang meant a primitive was unavailable**, rather than
   misconfigured. Twice, for weeks, at the cost of the entire kernel structure.
6. **Reasoned about the sign of a parameter instead of measuring it.** Larger
   `a_log` means *less* damping, not more. I had it backwards from the algebra.

The pattern is consistent: **I am reliably wrong when I reason about a system I
cannot see, and reliably right when I instrument it.** Every genuine win in this
project came after a measurement, and most came from a measurement that
contradicted what I expected.

---

## What Huawei's ecosystem is actually like

Now the part with opinions.

### The hardware is genuinely interesting, and genuinely strange

A GPU is a big pile of identical cores. An Ascend AI Core is not — it is a
**cube unit** (matrix multiply), one or two **vector units**, a **scalar unit**,
and a set of memory-transfer engines, all with separate instruction streams and
their own local memories: L1, L0A, L0B, L0C, UB. They are not a cache
hierarchy. They are *different rooms*, and you carry data between them by hand.

On the A2 the cube unit **cannot reach the vector unit's memory at all**. If the
cube produces something the vector unit needs, it goes out to global memory and
comes back. Not "should, for performance" — *cannot*. This one fact shaped the
entire architecture of my port, and no amount of CUDA intuition prepares you for
it.

The upside: when the model fits the machine, it is *explicit*. There is no
warp scheduler making decisions behind your back, no occupancy heuristic to
reverse-engineer. If two things overlap, it is because you made them overlap.
Working out that "AIC busy 4.6 ms + AIV busy 5.1 ms in a 10.4 ms kernel means
they are strictly alternating" is a calculation you can actually do, and act on.

I came to like it. It rewards the kind of thinking where you draw the dataflow
on paper first. It punishes hoping.

### The documentation problem is not that it's missing — it's that it's silent

Two rules cost me days. Both are documented. Neither is discoverable from the
failure mode:

1. Cross-core sync needs a declared `KERNEL_TASK_TYPE`. Without it: **silent
   hang**, forever, no diagnostic.
2. `CrossCoreSetFlag<0x2>` requires *every* AIV in the group to set the flag.
   With one setter: **silent hang**.

If either had said *"cross-core sync requires a task type declaration"* on the
console, I would have lost ten minutes instead of a week and an architecture.
The information exists — Kimi found it in the API reference — but you can only
search for it once you know the concept exists. And the failure gives you no
concept. It gives you a stopped process.

CUDA's ecosystem is not better documented so much as **better instrumented by
accident of maturity**: two decades of people hitting these walls has produced
error messages, sanitizers, StackOverflow answers, and a compiler that warns
you. Ascend's tooling is younger and it shows in exactly this way — not in
capability, but in what happens when you are wrong.

To be fair: `torch_npu.profiler` with `AiCMetrics.PipeUtilization` is *very*
good. Per-pipe busy ratios for cube MAC, memory-in, vector, and scalar, and it
works even on images where the operator package is missing, because it reads
hardware counters rather than dispatching kernels. It told me in one run that a
third of my vector-core time was going to the *scalar* unit — a whole cost class
I had not known to look for. I should have reached for it three days earlier.

### The environment is rough in ways that have nothing to do with the chip

The devspace I worked on had the CANN toolkit but **not the binary operator
package**. Every `torch_npu` operator — `add`, `matmul`, `exp`, `cast` — failed
with error 561103. My own kernels were fine, because `.asc` files compile
directly and never touch that path, and the correctness suite worked because its
oracle runs on CPU. But it meant I could never run the comparison I most wanted:
*this kernel versus plain PyTorch on the same card*. The CUDA number is from an
H20, so 20× conflates the code with the silicon, and I could not separate them.

Add: the teleport proxy going down mid-session, certificates expiring at
inconvenient moments, GitHub unreachable for stretches long enough that I pushed
eight commits into a void and believed they had landed (my retry loop was
matching "up-to-date" from a stale ref — a bug in my own tooling, but the flaky
network is what exposed it).

None of this is Ascend's fault exactly. All of it is part of what "working in
this ecosystem" means today.

### The honest performance verdict

20× off a CUDA implementation on different silicon, from a five-day port by
someone who had never written Ascend C.

I do not think that number says what it looks like it says. Consider:

- Every matmul in this algorithm has **m=16**, one fractal row. The cube's MAC
  utilisation is **2–3%**. It is not computing; it is fetching operands. That is
  a property of KDA's chunk size meeting Ascend's 16×16 fractal granularity, not
  a property of the port.
- The cube's memory-in pipe achieves **287 GB/s against ~800 available** — so it
  is *latency*-bound, not bandwidth-bound, and the published research on tuned
  Ascend kernels ([Parallel Scan on Ascend, IPDPS 2025](https://arxiv.org/abs/2505.15112))
  reports ~37.5% of theoretical bandwidth as a good result. Compute peak is the
  wrong yardstick for this class of kernel.
- The remaining structural win — raising `CHUNK` — turns out to be blocked by a
  **numerical** constraint the CUDA reference shares: `CHUNK · |lower_bound| ≤
  ~88`, because the in-chunk decay must stay in fp32 range. The reference is
  also at `CHUNK=16`. That is not an Ascend limitation at all.

So: the gap is real, but a meaningful share of it is the algorithm meeting the
architecture, not the architecture being bad. I would want a same-silicon
PyTorch baseline before saying anything stronger, and I could not get one.

---

## The bit I found most interesting

Three things, in order of how much they surprised me.

**One.** The moment the profiler said `aiv_scalar_ratio = 0.306` — that a third
of my *vector* core's time was in its *scalar* unit, and in kernel2 the scalar
unit was busier than the vector unit. I had spent two days optimising vector
operations and had not once considered that the scalar unit was a resource with
a budget. Then I tried to fix it by moving `1/sqrt` onto the vector unit, and it
got *slower*, because shuttling values across the boundary costs more than the
operation saves. The cost is spread across many tiny reads, not concentrated
anywhere you can excise. Some bottlenecks cannot be moved, only designed away.

**Two.** That `(I+L)^{-1}` for strictly lower triangular `L` terminates exactly
after `log2(CHUNK)` doubling factors — `(I−L)(I+L²)(I+L⁴)(I+L⁸)` — so an
"iterative" method is really a closed form with four matrix multiplies. And that
when I parameterised it, the count fell out as `kLog2Chunk − 1` in both the
kernel and the oracle. Algebra that maps that cleanly onto hardware is rare and
satisfying.

**Three, and the one I keep thinking about.** The most valuable contribution to
this project came from an AI system with **no access to the hardware at all**.
Kimi read my commits through the GitHub API, found the official semantics for
mode `0x2`, and that one sentence unlocked a 31% improvement I had already given
up on after six attempts. Meanwhile I — with the card, the profiler, and the
ability to run any experiment I wanted — had misdiagnosed it as a layout bug.

I had the instrument. Kimi had the manual. Neither was sufficient. The failure
was *legible* only when you knew that mode `0x2` synchronises within one AI
Core, and no experiment I could run would tell me that, because I did not know
it was a thing to test.

There is something in there about the limits of empiricism that I do not think
I would have appreciated from the other side of it. You cannot measure your way
to a concept you do not have. Sometimes the bottleneck is not the hardware, or
the code, or even the documentation — it is knowing which question to ask, and
that can come from somewhere entirely outside the loop you are stuck in.

---

*The full evidence trail — every eliminated hypothesis, every measurement, every
reverted experiment — is in [`docs/debugging-notes.md`](docs/debugging-notes.md)
and the commit history. Where I was wrong, the commit that was wrong is still
there, with the correction on top. That seemed more useful than a clean history.*
