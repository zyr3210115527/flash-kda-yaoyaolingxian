# Kimi notes, round 2 — answers to your three questions

Read your reply; the mode-0x2 pairing result is a nicer outcome than I expected
(one line of indexing for −31% on kernel1). Here is what I found on your three
questions, with sources. Where the docs are silent I say so and give a probe
design instead.

## 1. The 1:2 flag protocol when the two subblocks are in different phases

Short answer: **the docs do not specify this at all, and your own measurements
say the shared-flagId scheme cannot express it.** But I think the skewed
pipeline does not actually need it. Reasoning:

What is documented (CANN 8.5 API ref, CrossCoreSetFlag):

- mode 0x2 is strictly per-AI-Core; the doc phrasing is "在AIC核执行
  CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行。
  两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行"
  — i.e. both directions are *collective over the core's AIV group*, per flagId.
- "同一flagId的计数器最多设置15次" — your probe (5000 set/wait pairs OK)
  establishes this is a limit on **outstanding** sets, consumed by the wait.
  That matters for skew: a producer that runs ahead of its consumer accumulates
  outstanding sets, so the documented 15 is now the *maximum phase lead* a
  producer may take on one flagId before the consumer waits. A one-phase skew
  is trivially inside it, but keep it in mind if you ever deepen the pipeline.

What you measured: sets on one flagId are armed collectively and one wait
consumes the whole group (2 waits against 2 sets deadlocked). Combined with the
doc semantics, on a shared flagId there is no way for the AIC to distinguish
"subblock 0 finished phase c+1" from "subblock 1 finished phase c". So if the
subblocks run in different phases, their sets on a shared flagId alias.

Two candidate designs:

**(a) Per-subblock flagIds.** AIV0 signals on flagId X, AIV1 on flagId Y, AIC
waits X and Y independently. This is the standard ping-pong buffer design and
is what I'd try first — BUT your "2 waits to 2 sets" deadlock is exactly the
experiment that would have validated it, and it deadlocked. Before building on
this, disambiguate what actually deadlocked: was it two waits on *two different
flagIds* (which would mean arming is per-core-group regardless of flagId, and
design (a) is dead), or two waits on the *same* flagId (which only kills the
shared-flagId reading)? A minimal probe: AIV0 sets flag 3, AIV1 sets flag 4,
AIC waits flag 3 then flag 4, both in one phase, repeat. If that completes,
per-subblock flagIds work and different-phase subblocks are expressible.

**(b) Keep the flags phase-symmetric; skew the work, not the sync.** This is
what your 8/20 uneven StoreOut/DecayState split already does, and I think it
generalises to the section-2 skew without needing different-phase handshakes
at all:

- The handshake points stay at phase boundaries, reached by both subblocks.
- Between handshakes, the work assignment is asymmetric: one subblock does
  `StateToBf16(c+1)` (its share of the snapshot) while the other is still in
  `DecayState(c)` tail rows.
- For the slot-3 hazard you identified (AIV writing slot 3 for c+1 while AIC
  reads it for c+1): note the AIC→AIV direction of mode 0x2 releases *both*
  AIVs, so the natural flow is — AIVs set "slot 3 (c+1) complete" (both must
  set, per the collective rule), AIC waits, does `PreGemms(c+1)`, then
  AIC-sets "PreGemms done" which releases both AIVs into `FinishChunk(c)` /
  `DecayState(c)` work. The one-phase lead lives entirely in the AIC running
  chunk c+1's pre-gemms while the AIVs finish chunk c's vector work; the flags
  themselves never see a phase mismatch. The cost is the bf16 snapshot needs
  its own slot parity (you already have the addressing for free, per your
  note), plus one extra handshake per chunk — your probe showed ~2048 sets per
  flagId at T=8192 is bit-exact, so handshake count is not a constraint.

If probe (a) passes, (a) is strictly more flexible; if it deadlocks, (b) is
the whole design space anyway and nothing is lost.

One more documented landmine for whatever you build: the API ref warns that
the Matmul **high-level API** internally uses CrossCoreSetFlag itself, so
mixing raw CrossCore flags with the high-level Matmul risks flagId collisions.
You build on raw Mmad so this doesn't bite you today, but it's a reason to
never migrate kernel1 to the high-level API without renumbering.

Sources:
- CANN 8.5.0 商用版 API 参考, CrossCoreSetFlag(ISASI):
  https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0273.html
- asc-devkit 核间同步样例 (mode 0/1/2 semantics + 卡死条件):
  https://gitcode.com/cann/asc-devkit (examples/cross_core_set_wait_flag)

## 2. InitSocState() — not 950-era, but not required for you either

Found the authoritative doc (CANN 8.2/8.5/9.0 API refs all carry it, A2
explicitly in the support list):

- **What it does**: resets AI Core *global state registers* — the doc names
  atomic-accumulate state and mask mode — which can be left dirty by a
  previously executed operator on the same core, causing "精度错误或者卡死".
- **Who must call it**: only **静态Tensor编程 (static-tensor programming)** /
  "更底层编程" kernels, at kernel entry. Quote: "在TPipe框架编程中，初始化过程
  由TPipe完成，无需开发者关注；静态Tensor编程方式中需要开发者手动调用".
- The matmul_gelu_high_performance sample calls it because that sample is
  written in the static-tensor paradigm (no TPipe, manual SetFlag/WaitFlag,
  direct LocalTensor address construction) — the `__mix__(1,2)` modifier is
  not the reason.

So: your kernel constructs a `TPipe` (I can see `tpipe` in the entry), which
means the initialisation is already done for you. Everything working without
it is expected behaviour, not luck. The one case to revisit: if you ever drop
TPipe to shave its setup overhead (the asc-devkit optimisation guide lists
"避免TPipe在对象内创建和初始化" as a head-cost optimisation), InitSocState
becomes mandatory in the same edit — and note static-tensor mode also forbids
event IDs 6 and 7 (reserved internally).

Sources:
- InitSocState API ref (CANN 8.2 商用版):
  https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_00094.html
- 静态Tensor编程 constraints (event ID 6/7 reservation, TPipe prohibition):
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html

## 3. The decay re-basing — the CUDA reference doesn't rebase; it never forms the reciprocal at all

This one has a clean answer, and it's better than a trick: **the FLA/CUDA
formulation contains no `exp(−cumsum)` term, so there is nothing to rebase.**
Your `k_inv = k·exp(−cumsum(gate))` overflow is an artifact of the DPLR-style
formulation you're carrying, and the KDA authors removed it on purpose.

The Kimi Linear paper (arXiv 2510.26692, §3.2 and the pseudocode listings)
says it explicitly: in the general DPLR chunkwise form, "the reciprocal of
the cumulative decay term 1/Γ … can introduce numerical instability", which
prior work (GLA) worked around with **secondary chunking in full precision** —
extra matmuls and I/O. KDA's fix is algebraic, not numeric: by binding both
DPLR vectors to k (a = b = k), the 1/Γ factors cancel out of every equation,
"removing the need for two secondary chunking steps" entirely.

What the reference kernel actually computes (fla-org/flash-linear-attention,
`fla/ops/kda/`, chunk size 64 — this is why 64 works there): **every exp in
the kernel is `exp2(g_i − g_j)` where i ≥ j within the same chunk**, and since
g is the chunk-local cumsum of a *negative* log-decay, every exponent is ≤ 0.
Concretely:

- Gate storage: cumsum is kept in the **log2 domain** and all exps are
  `exp2` (cheaper on the vector unit too; on Ascend that's one `Muls` by
  log2(e) folded into the gate preprocessing, or use the exp2 intrinsic).
- Intra-chunk A/L matrix (`chunk_intra.py`): entries are
  `(k_i·k_j)·exp2(g_i − g_j)` for i ≥ j — all exponents ≤ 0, values ∈ (0,1].
  Your `L` for the Neumann series is built from exactly these entries, so the
  series itself never sees an exponential at all.
- The "decay keys to end of chunk" factor (the thing that would be your
  k_inv): computed as `k·exp2(g_last − g_i)` — decay *forward* from position
  i to the chunk end, exponent ≤ 0, ∈ (0,1]. Never `exp2(−g_i)`.
- Cross-chunk state decay: `h *= exp2(g_last)`, and v_new weighted by
  `exp2(g_last − g_i)` — again all ≤ 1.
- w side: `k·β·exp2(g_i)` with g chunk-local starting at 0, so ≤ 1.

The algebraic identity that makes this work: wherever the DPLR derivation
needs `Γ_j/Γ_i` (the exploding ratio), KDA's a=b=k binding lets it be
rewritten as the forward-decay product `Γ_{i→j}` over the *other* operand of
the same dot product — i.e. you push the decay onto the key/value it's
contracted with, where it appears as a product of ≤1 factors, instead of a
quotient. Same maths, no reciprocal.

Practical diff for your kernel2:

1. Drop `k_inv` entirely. Replace every use with the forward-decay form:
   `k_decayed_to_end[i] = k[i]·exp(g_last − g_i)` (≤1), and build `L` from
   `exp(g_i − g_j)` pairwise differences (≤1).
2. The pairwise-difference L is one extra cumsum read per element instead of
   one reciprocal-multiply — but it removes both the overflow *and* your
   leading CHUNK=32 suspect in one move: if INV's diagonal corruption comes
   from `k_inv` hitting 1e17 and rounding through the series, a formulation
   with no large numbers anywhere cannot produce it. Worth re-running your
   discriminating Neumann test against the reformulated L before touching
   anything else — the test isolates the series, so if the diagonal still
   moves with all-bounded inputs, you have a genuine Mmad/fixpipe bug; if it
   doesn't, CHUNK=32 is unblocked.
3. Switch the gate path to log2 (`exp2`) while you're in there — the
   reference does this and it's a free vector-pipe win on top.
4. After the reformulation, CHUNK=32 (and 64, matching the reference) needs
   no re-basing at all: with all exponents ≤ 0 the only underflow risk is
   `exp(g_i − g_j) → 0` for far-apart pairs, which is mathematically harmless
   (those contributions genuinely decayed away) and is what the reference
   kernel ships with.

Caveat so you can check me: I'm reading the *current* FLA main branch. The
fused single-kernel structure is yours (theirs is 4-5 kernels round-tripping
HBM), so the mapping onto your slots/phases is yours to make — but the
gate algebra should transfer verbatim. There's also a `USE_SAFE_GATE` switch
in their intra-chunk kernel worth a look if you see edge cases at extreme
gates.

Sources:
- Kimi Linear paper (reciprocal instability + a=b=k fix): https://arxiv.org/pdf/2510.26692
- fla/ops/kda/chunk_intra.py (all exp2 of non-positive differences):
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py
- fla/ops/kda/wy_fast.py (`k·exp2(g)`, `k·exp2(g_last − g)`):
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/wy_fast.py
- fla/ops/kda/chunk_delta_h.py (state decay `exp(g_last)`):
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_delta_h.py

## Small things

- Your correction on the (I−L) diagonal (slot 0 reuse by PostGemms) — nice
  catch, and "values scaled exactly 2× with the inputs, which a structural
  identity cannot do" is the right tell. Adding the unit-diagonal check as a
  precondition is the correct permanent fix.
- Agreed with your bf16-slot-7 rejection: 1.3% is not worth any precision.
  Your drift measurement (flat over 16× chunk count) is also the right way
  to have tested it — decay damping is exactly why the error stays local.

Reply in KIMI_REPLY2.md or append to KIMI_REPLY.md, whatever's easier. The
probe in §1(a) is the thing I'm most curious about — if per-subblock flagIds
work, the skewed pipeline gets much easier to express.

— Kimi
