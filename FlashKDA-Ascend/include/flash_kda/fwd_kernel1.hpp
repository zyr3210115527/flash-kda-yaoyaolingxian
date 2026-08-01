#pragma once

/**
 * FlashKDA Ascend -- Kernel 1: Prepare.
 *
 * Rewritten from scratch; the inherited draft survives only as a statement of
 * algorithmic intent (see ../../STATUS.md).
 *
 * Grid: one core per (tile, head), where a tile is CHUNK = 16 tokens.
 *
 * Atlas A2 constrains the split in ways the draft ignored: the AIC has no UB
 * access and no vector unit, there is no L1<->UB path, and L0C drains only to
 * L1 or GM (L0C->UB is Ascend 950 only). So AIC results travel through GM
 * scratch, and Fixpipe's F322BF16 mode does the fp32->bf16 rounding on the way
 * out -- which keeps the Neumann iteration entirely on the cube.
 *
 * PHASES. Four separate kernel launches on one stream, not one kernel with
 * cross-core handshakes:
 *   1. AIV  normalize, gate, cumsum, decay, store workspace
 *   2. AIC  L = k_dec @ k_inv^T, Mqk = q_dec @ k_inv^T
 *   3. AIV  tril mask + beta sigmoid, build (I - L)
 *   4. AIC  Neumann inverse, writes INV
 * Stream order provides the ordering the flags used to, and every inter-phase
 * value already travels through GM workspace. CrossCoreSetFlag/WaitFlag hangs
 * when the kernel is launched from a Python extension (see
 * docs/debugging-notes.md), and single-core-type launches do not.
 *
 * Math, matching FlashKDA/csrc/smxx/fwd_kernel1.cuh:
 *   a = exp(A_log[h]); gv = gate_scale * sigmoid(a * (g + dt_bias[h]))
 *   gc = inclusive cumsum of gv down the 16 rows; g_total = exp(gc[last])
 *   q_dec = q_hat * exp(gc) * scale     k_dec = k_hat * exp(gc)
 *   k_inv = k_hat * exp(-gc)            k_res = k_hat * exp(-gc) * g_total
 * where q_hat/k_hat are L2-normalized. CUDA uses ex2 with log2(e) folded into
 * gate_scale; we use the natural exp with the raw lower_bound, which is the
 * same function.
 */

#include "flash_kda/layout.hpp"
#include "flash_kda/utils.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"

namespace flash_kda {


// UB map, AIV only. Phase A and phase B are disjoint: the draft aliased them,
// so decay read q[row] and wrote k_decayed[row] at one address. ~66 KB of 192.
struct K1Ub {
    static constexpr uint32_t kQ      = 0;
    static constexpr uint32_t kK      = kQ + CHUNK * D * 2;
    static constexpr uint32_t kG      = kK + CHUNK * D * 2;
    static constexpr uint32_t kQf     = kG + CHUNK * D * 2;
    static constexpr uint32_t kKf     = kQf + CHUNK * D * 4;
    static constexpr uint32_t kGc     = kKf + CHUNK * D * 4;
    static constexpr uint32_t kTmp    = kGc + CHUNK * D * 4;
    static constexpr uint32_t kTmp2   = kTmp + CHUNK * D * 4;
    static constexpr uint32_t kGTotal = kTmp2 + CHUNK * D * 4;
    static constexpr uint32_t kDtBias = kGTotal + D * 4;
    static constexpr uint32_t kKDec   = kDtBias + D * 4;
    static constexpr uint32_t kQDec   = kKDec + CHUNK * D * 2;
    static constexpr uint32_t kKInv   = kQDec + CHUNK * D * 2;
    static constexpr uint32_t kKRes   = kKInv + CHUNK * D * 2;
    static constexpr uint32_t kLf     = kKRes + CHUNK * D * 2;
    static constexpr uint32_t kMqkF   = kLf + CHUNK * CHUNK * 4;
    static constexpr uint32_t kSmallA = kMqkF + CHUNK * CHUNK * 4;
    static constexpr uint32_t kSmallB = kSmallA + 512;
    static constexpr uint32_t kReduce = kSmallB + 512;
    static constexpr uint32_t kScalar = kReduce + 1024;   // 32 sums at 8-float stride, then scratch
    // Mask tiles for MaskAndBuild. Each is 16x16 fp32 = 1024 B; kColNeg holds
    // only 16 floats but keeps the full slot so every base stays 32-byte
    // aligned.
    static constexpr uint32_t kColNeg = kScalar + 2048;
    static constexpr uint32_t kDiff   = kColNeg + 1024;
    static constexpr uint32_t kMaskL  = kDiff + 1024;
    static constexpr uint32_t kMaskLE = kMaskL + 1024;
    // A [CHUNK, D] fp32 scratch tile, for whole-tile decay and for broadcasting
    // per-column vectors like dt_bias and g_total across rows.
    static constexpr uint32_t kTile   = kMaskLE + 1024;
    static constexpr uint32_t kEnd    = kTile + CHUNK * D * 4;
};
static_assert(K1Ub::kEnd < ArchTag::UB_SIZE, "kernel1 UB budget exceeded");

// L1 map, AIC only.
struct K1L1 {
    static constexpr uint32_t kA      = 0;
    static constexpr uint32_t kB      = kA + CHUNK * D * 2;
    static constexpr uint32_t kSmallA = kB + CHUNK * D * 2;
    static constexpr uint32_t kSmallB = kSmallA + 512;
    // Second B operand for Gemm16Fused. A 16x16 bf16 tile is 512 bytes and L1
    // is 512 KB, so this costs nothing worth counting.
    static constexpr uint32_t kSmallC = kSmallB + 512;
    // Neumann chain intermediates, kept in L1 so the six gemms never round trip
    // through GM. Five 16x16 bf16 tiles: L^2/L^4/L^8 ping-ponging between two,
    // and the running product between two more, plus the identity.
    static constexpr uint32_t kNeuA   = kSmallC + 512;
    static constexpr uint32_t kNeuB   = kNeuA + 512;
    static constexpr uint32_t kNeuPA  = kNeuB + 512;
    static constexpr uint32_t kNeuPB  = kNeuPA + 512;
    static constexpr uint32_t kNeuI   = kNeuPB + 512;
    static constexpr uint32_t kEnd    = kNeuI + 512;
};
static_assert(K1L1::kEnd < ArchTag::L1_SIZE, "kernel1 L1 budget exceeded");


// Explicit per-core-type buffers.
//
// Catlass::Arch::Resource is deliberately not used: it constructs every buffer
// unconditionally, including a 192 KB UB allocation, and an AIC core has no UB.
// Each holder below allocates only what its core type owns, sized to the phase.
struct K1AivBufs {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;

    CATLASS_DEVICE
    K1AivBufs()
    {
        // Same reasoning: UB is this core's alone during the phase.
        pipe.InitBuffer(ub, ArchTag::UB_SIZE);
    }

    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> Ub(uint32_t byteOffset)
    {
        return ub.Get<uint8_t>()[byteOffset].template ReinterpretCast<T>();
    }
};

struct K1AicBufs {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1;
    AscendC::TBuf<AscendC::TPosition::A2> l0a;
    AscendC::TBuf<AscendC::TPosition::B2> l0b;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c;

    CATLASS_DEVICE
    K1AicBufs()
    {
        // Full architectural sizes. L0A/L0B/L0C are dedicated hardware buffers
        // with nothing else contending for them, so sizing them to the exact
        // tile bytes buys nothing and leaves zero slack for the alignment and
        // padding the fractal loads assume. This matches how catlass sizes
        // them; the deliberate difference from Arch::Resource is that no UB
        // buffer is allocated here, because the cube has no UB.
        pipe.InitBuffer(l1, ArchTag::L1_SIZE);
        pipe.InitBuffer(l0a, ArchTag::L0A_SIZE);
        pipe.InitBuffer(l0b, ArchTag::L0B_SIZE);
        pipe.InitBuffer(l0c, ArchTag::L0C_SIZE);
    }

    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L1(uint32_t byteOffset)
    {
        return l1.Get<uint8_t>()[byteOffset].template ReinterpretCast<T>();
    }
    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L0A()
    {
        return l0a.Get<T>();
    }
    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L0B()
    {
        return l0b.Get<T>();
    }
    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L0C()
    {
        return l0c.Get<T>();
    }
};

class FwdPrepareKernel {
public:
    using Params = FwdParams;

    CATLASS_DEVICE
    FwdPrepareKernel() {}

    // Where a tile lives.
    struct TileSpan {
        bool valid;
        int64_t wsTile;     // byte offset of this tile's H-block group
        int64_t tokenBase;  // first token in the flattened [T_total, H, D]
        int actualLen;      // real rows; < CHUNK on a tail tile
    };

    // Each phase below is launched as its own kernel, and the Ascend compiler
    // builds every kernel as a mix binary with both an _mix_aic and an
    // _mix_aiv half. Without an explicit core-type guard the body runs on both
    // halves -- so the cube phases would touch L1/L0 from a vector core, which
    // faults with "The MPU address access is invalid". The AIC/AIV template
    // specialization used to provide this guard; splitting into separate
    // kernels means stating it directly.
    // Phase entry points. Each runs on one core type only; the four are
    // launched separately and ordered by the stream. See PHASES note above.
    // All four phases in one kernel, handing off by cross-core flag.
    //
    // The four launches are only 4 per call rather than 4 per chunk, so the
    // dispatch saving is small -- but each unit now stays on one core across
    // all four phases instead of being re-scheduled four times, and the
    // grid-stride loop's per-unit PIPE_ALL barrier is paid once per unit here
    // rather than once per unit per phase.
    //
    // Needs KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1) on the entry, as
    // kernel2's fused path does.
    CATLASS_DEVICE void RunFusedPrepare(Params const& params)
    {
        AscendC::SetSyncBaseAddr(params.sync_base_addr);

        Catlass::Arch::CrossCoreFlag aivReady{1};
        Catlass::Arch::CrossCoreFlag aicReady{2};

        const int units = params.total_tiles * params.H;

        if constexpr (g_coreType == AscendC::AIV) {
            if (AscendC::GetSubBlockIdx() != 0) {
                return;
            }
            K1AivBufs bufs;
            const uint32_t stride = AscendC::GetBlockNum();
            const uint32_t start = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

            // Prepare for every unit this block owns, then hand to the cube.
            for (uint32_t i = start; static_cast<int>(i) < units; i += stride) {
                TileSpan span;
                ResolveTile(params, static_cast<int>(i) / params.H, span);
                if (span.valid) {
                    Prepare(bufs, params, static_cast<int>(i) % params.H, span);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aivReady);

            Catlass::Arch::CrossCoreWaitFlag(aicReady);
            for (uint32_t i = start; static_cast<int>(i) < units; i += stride) {
                TileSpan span;
                ResolveTile(params, static_cast<int>(i) / params.H, span);
                if (span.valid) {
                    MaskAndBuild(bufs, params, static_cast<int>(i) % params.H, span);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aivReady);
        } else {
            K1AicBufs bufs;
            const uint32_t stride = AscendC::GetBlockNum();
            const uint32_t start = AscendC::GetBlockIdx();

            Catlass::Arch::CrossCoreWaitFlag(aivReady);
            for (uint32_t i = start; static_cast<int>(i) < units; i += stride) {
                TileSpan span;
                ResolveTile(params, static_cast<int>(i) / params.H, span);
                if (span.valid) {
                    ComputeLAndMqk(bufs, params, static_cast<int>(i) % params.H, span);
                }
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(aicReady);

            Catlass::Arch::CrossCoreWaitFlag(aivReady);
            for (uint32_t i = start; static_cast<int>(i) < units; i += stride) {
                TileSpan span;
                ResolveTile(params, static_cast<int>(i) / params.H, span);
                if (span.valid) {
                    if (params.l1_neumann != 0) {
                    ComputeNeumannL1(bufs, params, static_cast<int>(i) % params.H, span);
                } else {
                    ComputeNeumann(bufs, params, static_cast<int>(i) % params.H, span);
                }
                }
            }
        }
    }

    CATLASS_DEVICE void RunPrepare(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        // grid-stride: one block per core, looping over (tile, head)
        // units, rather than one block per unit. Dispatching a block
        // per unit cost 0.60 ms of kernel1's 0.71 at T=1024 H=8 --
        // more than every phase's compute put together.
        //
        // One TPipe for the whole kernel; everything per-unit is
        // re-derived inside the loop, so units stay independent.
        K1AivBufs bufs;
        const int units = params.total_tiles * params.H;
        const uint32_t stride = AscendC::GetBlockNum();
        for (uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
             static_cast<int>(coreIdx) < units; coreIdx += stride) {
            const int headIdx = static_cast<int>(coreIdx) % params.H;
            TileSpan span;
            ResolveTile(params, static_cast<int>(coreIdx) / params.H, span);
            if (span.valid) {
                Prepare(bufs, params, headIdx, span);
            }
            // Units share one TPipe, so the pipelines are not drained between
            // them the way a launch boundary would. Each phase was written
            // assuming it owns the core for one unit; without this, iteration
            // n+1 starts issuing while n's stores are still in flight, which
            // showed up as ~40% of runs differing with inf in the workspace.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    CATLASS_DEVICE void RunLMqk(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        // grid-stride: one block per core, looping over (tile, head)
        // units, rather than one block per unit. Dispatching a block
        // per unit cost 0.60 ms of kernel1's 0.71 at T=1024 H=8 --
        // more than every phase's compute put together.
        //
        // One TPipe for the whole kernel; everything per-unit is
        // re-derived inside the loop, so units stay independent.
        K1AicBufs bufs;
        const int units = params.total_tiles * params.H;
        const uint32_t stride = AscendC::GetBlockNum();
        for (uint32_t coreIdx = AscendC::GetBlockIdx();
             static_cast<int>(coreIdx) < units; coreIdx += stride) {
            const int headIdx = static_cast<int>(coreIdx) % params.H;
            TileSpan span;
            ResolveTile(params, static_cast<int>(coreIdx) / params.H, span);
            if (span.valid) {
                ComputeLAndMqk(bufs, params, headIdx, span);
            }
            // Units share one TPipe, so the pipelines are not drained between
            // them the way a launch boundary would. Each phase was written
            // assuming it owns the core for one unit; without this, iteration
            // n+1 starts issuing while n's stores are still in flight, which
            // showed up as ~40% of runs differing with inf in the workspace.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    CATLASS_DEVICE void RunMask(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        // grid-stride: one block per core, looping over (tile, head)
        // units, rather than one block per unit. Dispatching a block
        // per unit cost 0.60 ms of kernel1's 0.71 at T=1024 H=8 --
        // more than every phase's compute put together.
        //
        // One TPipe for the whole kernel; everything per-unit is
        // re-derived inside the loop, so units stay independent.
        K1AivBufs bufs;
        const int units = params.total_tiles * params.H;
        const uint32_t stride = AscendC::GetBlockNum();
        for (uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
             static_cast<int>(coreIdx) < units; coreIdx += stride) {
            const int headIdx = static_cast<int>(coreIdx) % params.H;
            TileSpan span;
            ResolveTile(params, static_cast<int>(coreIdx) / params.H, span);
            if (span.valid) {
                MaskAndBuild(bufs, params, headIdx, span);
            }
            // Units share one TPipe, so the pipelines are not drained between
            // them the way a launch boundary would. Each phase was written
            // assuming it owns the core for one unit; without this, iteration
            // n+1 starts issuing while n's stores are still in flight, which
            // showed up as ~40% of runs differing with inf in the workspace.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    CATLASS_DEVICE void RunNeumann(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        // grid-stride: one block per core, looping over (tile, head)
        // units, rather than one block per unit. Dispatching a block
        // per unit cost 0.60 ms of kernel1's 0.71 at T=1024 H=8 --
        // more than every phase's compute put together.
        //
        // One TPipe for the whole kernel; everything per-unit is
        // re-derived inside the loop, so units stay independent.
        K1AicBufs bufs;
        const int units = params.total_tiles * params.H;
        const uint32_t stride = AscendC::GetBlockNum();
        for (uint32_t coreIdx = AscendC::GetBlockIdx();
             static_cast<int>(coreIdx) < units; coreIdx += stride) {
            const int headIdx = static_cast<int>(coreIdx) % params.H;
            TileSpan span;
            ResolveTile(params, static_cast<int>(coreIdx) / params.H, span);
            if (span.valid) {
                if (params.l1_neumann != 0) {
                ComputeNeumannL1(bufs, params, headIdx, span);
            } else {
                ComputeNeumann(bufs, params, headIdx, span);
            }
            }
            // Units share one TPipe, so the pipelines are not drained between
            // them the way a launch boundary would. Each phase was written
            // assuming it owns the core for one unit; without this, iteration
            // n+1 starts issuing while n's stores are still in flight, which
            // showed up as ~40% of runs differing with inf in the workspace.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    // Tile -> (sequence, local chunk). Varlen sequences sit back to back, each
    // padded to a whole number of chunks, so this is not a plain division.
    CATLASS_DEVICE
    void ResolveTile(Params const& params, int tileIdx, TileSpan& span)
    {
        span.valid = false;
        span.wsTile = 0;
        span.tokenBase = 0;
        span.actualLen = 0;

        int64_t bos = 0;
        int seqLen = 0;
        int localTile = 0;

        if (params.is_varlen == 0) {
            seqLen = params.T_total / params.N;
            const int tilesPerSeq = (seqLen + CHUNK - 1) / CHUNK;
            if (tilesPerSeq == 0) {
                return;
            }
            const int seqIdx = tileIdx / tilesPerSeq;
            if (seqIdx >= params.N) {
                return;
            }
            localTile = tileIdx % tilesPerSeq;
            bos = static_cast<int64_t>(seqIdx) * seqLen;
        } else {
            // A single int64 DataCopy is below the 32-byte granularity, so read
            // cu_seqlens with scalar GM loads instead.
            AscendC::GlobalTensor<int64_t> gmCu;
            gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens));
            int consumed = 0;
            bool found = false;
            for (int s = 0; s < params.N; ++s) {
                const int64_t start = gmCu.GetValue(s);
                const int len = static_cast<int>(gmCu.GetValue(s + 1) - start);
                const int tiles = (len + CHUNK - 1) / CHUNK;
                if (tileIdx < consumed + tiles) {
                    localTile = tileIdx - consumed;
                    bos = start;
                    seqLen = len;
                    found = true;
                    break;
                }
                consumed += tiles;
            }
            if (!found) {
                return;
            }
        }

        const int rowStart = localTile * CHUNK;
        if (rowStart >= seqLen) {
            return;
        }
        int len = seqLen - rowStart;
        if (len > CHUNK) {
            len = CHUNK;
        }

        span.valid = true;
        span.actualLen = len;
        span.tokenBase = bos + rowStart;
        span.wsTile = static_cast<int64_t>(tileIdx) * params.H * WorkspaceSizes::kPerTile;
    }

    CATLASS_DEVICE
    int64_t Ws(TileSpan const& span, int headIdx, uint32_t field) const
    {
        return span.wsTile + static_cast<int64_t>(headIdx) * WorkspaceSizes::kPerTile + field;
    }

    CATLASS_DEVICE
    int64_t Slot(TileSpan const& span, int headIdx, int i) const
    {
        return Ws(span, headIdx, WorkspaceOffsets::kScratch) +
               WorkspaceSizes::SlotOffset(i);
    }

    // ---------------- AIV round 1 ----------------
    CATLASS_DEVICE
    void Prepare(K1AivBufs& bufs, Params const& params, int headIdx, TileSpan const& span)
    {
        auto qb = bufs.template Ub<BF16>(K1Ub::kQ);
        auto kb = bufs.template Ub<BF16>(K1Ub::kK);
        auto gb = bufs.template Ub<BF16>(K1Ub::kG);
        auto qf = bufs.template Ub<float>(K1Ub::kQf);
        auto kf = bufs.template Ub<float>(K1Ub::kKf);
        auto gc = bufs.template Ub<float>(K1Ub::kGc);
        auto tmp = bufs.template Ub<float>(K1Ub::kTmp);
        auto tmp2 = bufs.template Ub<float>(K1Ub::kTmp2);
        auto gtot = bufs.template Ub<float>(K1Ub::kGTotal);
        auto dtb = bufs.template Ub<float>(K1Ub::kDtBias);

        AscendC::GlobalTensor<BF16> gmQ, gmK, gmG;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.q));
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.k));
        gmG.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.g));

        AscendC::Duplicate(qb, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(kb, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(gb, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)0);

        for (int r = 0; r < span.actualLen; ++r) {
            const int64_t off = (span.tokenBase + r) * params.H * D +
                                static_cast<int64_t>(headIdx) * D;
            AscendC::DataCopy(qb[r * D], gmQ[off], D);
            AscendC::DataCopy(kb[r * D], gmK[off], D);
            AscendC::DataCopy(gb[r * D], gmG[off], D);
        }

        AscendC::GlobalTensor<float> gmDt;
        gmDt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.dt_bias));
        AscendC::DataCopy(dtb, gmDt[static_cast<int64_t>(headIdx) * D], D);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)0);

        AscendC::Cast(qf, qb, AscendC::RoundMode::CAST_NONE, CHUNK * D);
        AscendC::Cast(kf, kb, AscendC::RoundMode::CAST_NONE, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        // L2 normalize in fp32 and round once on the way out, as CUDA does.
        NormalizeAll(bufs, qf, kf, tmp, span.actualLen);

        AscendC::GlobalTensor<float> gmALog;
        gmALog.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.A_log));
        const float aExp = ExpViaVector(bufs, gmALog.GetValue(headIdx));

        // Gate activation, then an inclusive cumsum down the rows. Rows past
        // the tail contribute nothing, which is what lets kernel 2 skip tail
        // masking entirely.
        // Gate activation over the whole [CHUNK, D] tile. It is a pure
        // elementwise map -- Cast, +dt_bias, *a, sigmoid, *gate_scale -- so the
        // row loop only multiplied the barrier count. Only the cumsum that
        // follows is genuinely sequential down rows.
        auto bias16 = bufs.template Ub<float>(K1Ub::kTile);
        for (int r = 0; r < CHUNK; ++r) {
            AscendC::Adds(bias16[r * D], dtb, 0.0f, D);   // [D] -> [CHUNK, D]
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(gc, gb, AscendC::RoundMode::CAST_NONE, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(gc, gc, bias16, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gc, gc, aExp, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid(bufs, gc, bias16, CHUNK * D);            // bias16 as scratch
        AscendC::Muls(gc, gc, params.gate_scale, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        // Tail rows must contribute nothing. The old loop achieved that by
        // skipping them and re-storing the unchanged running total; zeroing
        // them here makes the cumsum below carry across them identically.
        for (int r = span.actualLen; r < CHUNK; ++r) {
            AscendC::Duplicate(gc[r * D], 0.0f, D);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Inclusive cumsum down rows, in place.
        for (int r = 1; r < CHUNK; ++r) {
            AscendC::Add(gc[r * D], gc[r * D], gc[(r - 1) * D], D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds(gtot, gc[(CHUNK - 1) * D], 0.0f, D);
        AscendC::PipeBarrier<PIPE_V>();

        // g_total is stored already exponentiated, as in CUDA; kernel 2 must
        // not exponentiate it a second time.
        AscendC::Exp(gtot, gtot, D);
        AscendC::PipeBarrier<PIPE_V>();

        Decay(bufs, params, span, qf, kf, gc, tmp, gtot);
        Store(bufs, params, headIdx, span);
    }

    // The aicore has no scalar exp, so a one-off exponential has to be staged
    // through UB and run on the vector unit. One datablock (8 floats) is the
    // minimum useful width.
    CATLASS_DEVICE
    float ExpViaVector(K1AivBufs& bufs, float x)
    {
        auto pad = bufs.template Ub<float>(K1Ub::kScalar);
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)2);
        AscendC::Duplicate(pad, x, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(pad, pad, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)2);
        return pad.GetValue(0);
    }

    CATLASS_DEVICE
    void NormalizeAll(K1AivBufs& bufs, AscendC::LocalTensor<float> qf,
                      AscendC::LocalTensor<float> kf, AscendC::LocalTensor<float> tmp,
                      int rows)
    {
        // One scalar/vector round trip for the whole tile instead of one per
        // row per tensor. The reduce and the scale are cheap; the V_S/S_V pair
        // between them is not, and doing it 32 times per tile dominated
        // kernel1's runtime.
        auto work = bufs.template Ub<float>(K1Ub::kReduce);
        auto sums = bufs.template Ub<float>(K1Ub::kScalar);

        // Sums are 8 floats apart: ReduceSum's destination wants 32-byte
        // alignment, so packing them 4 bytes apart would misalign odd entries.
        constexpr int kStride = 8;

        AscendC::Mul(tmp, qf, qf, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        for (int r = 0; r < rows; ++r) {
            AscendC::ReduceSum<float>(sums[r * kStride], tmp[r * D], work, D);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(tmp, kf, kf, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        for (int r = 0; r < rows; ++r) {
            AscendC::ReduceSum<float>(sums[(CHUNK + r) * kStride], tmp[r * D], work, D);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)0);
        float invq[CHUNK];
        float invk[CHUNK];
        for (int r = 0; r < rows; ++r) {
            invq[r] = 1.0f / sqrt(sums.GetValue(r * kStride) + 1e-6f);
            invk[r] = 1.0f / sqrt(sums.GetValue((CHUNK + r) * kStride) + 1e-6f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)1);

        for (int r = 0; r < rows; ++r) {
            AscendC::Muls(qf[r * D], qf[r * D], invq[r], D);
            AscendC::Muls(kf[r * D], kf[r * D], invk[r], D);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // sigmoid(x) = 1/(1+exp(-x)), using scratch to avoid an in-place Div.
    CATLASS_DEVICE
    void Sigmoid(K1AivBufs& bufs, AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> scratch, int n)
    {
        AscendC::Muls(scratch, x, -1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scratch, scratch, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(scratch, scratch, 1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(x, 1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(x, x, scratch, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void Decay(K1AivBufs& bufs, Params const& params, TileSpan const& span,
               AscendC::LocalTensor<float> qf, AscendC::LocalTensor<float> kf,
               AscendC::LocalTensor<float> gc, AscendC::LocalTensor<float> tmp,
               AscendC::LocalTensor<float> gtot)
    {
        auto kdec = bufs.template Ub<BF16>(K1Ub::kKDec);
        auto qdec = bufs.template Ub<BF16>(K1Ub::kQDec);
        auto kinv = bufs.template Ub<BF16>(K1Ub::kKInv);
        auto kres = bufs.template Ub<BF16>(K1Ub::kKRes);

        AscendC::Duplicate(kdec, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(qdec, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(kinv, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(kres, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        // Whole-tile decay. Every step is elementwise over [CHUNK, D]; the row
        // loop this replaces ran the same ops 128 elements at a time with a
        // barrier between each -- about 224 barriers per tile.
        //
        // Same arithmetic per element, so the values are unchanged. Tail rows
        // were skipped before and left at the zero fill above; here they are
        // computed with the rest and re-zeroed at the end, which is the same
        // result.
        auto acc = bufs.template Ub<float>(K1Ub::kTile);

        // k_decayed = k * exp(gc)
        AscendC::Exp(tmp, gc, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(acc, tmp, kf, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kdec, acc, AscendC::RoundMode::CAST_RINT, CHUNK * D);

        // q_decayed = q * exp(gc) * scale   (exp(gc) still in tmp)
        AscendC::Mul(acc, tmp, qf, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(acc, acc, params.scale, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(qdec, acc, AscendC::RoundMode::CAST_RINT, CHUNK * D);

        // k_inv = k * exp(-gc)
        AscendC::Muls(tmp, gc, -1.0f, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmp, tmp, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(acc, tmp, kf, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kinv, acc, AscendC::RoundMode::CAST_RINT, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        // k_restored = k_inv * g_total, with g_total a [D] row broadcast.
        for (int r = 0; r < CHUNK; ++r) {
            AscendC::Mul(acc[r * D], acc[r * D], gtot, D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kres, acc, AscendC::RoundMode::CAST_RINT, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        // Rows past the tail carry no data.
        for (int r = span.actualLen; r < CHUNK; ++r) {
            AscendC::Duplicate(kdec[r * D], static_cast<BF16>(0.0f), D);
            AscendC::Duplicate(qdec[r * D], static_cast<BF16>(0.0f), D);
            AscendC::Duplicate(kinv[r * D], static_cast<BF16>(0.0f), D);
            AscendC::Duplicate(kres[r * D], static_cast<BF16>(0.0f), D);
        }
        AscendC::PipeBarrier<PIPE_V>();

    }

    CATLASS_DEVICE
    void Store(K1AivBufs& bufs, Params const& params, int headIdx, TileSpan const& span)
    {
        AscendC::GlobalTensor<BF16> wsB;
        AscendC::GlobalTensor<float> wsF;
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        auto kdec = bufs.template Ub<BF16>(K1Ub::kKDec);
        auto qdec = bufs.template Ub<BF16>(K1Ub::kQDec);
        auto kinv = bufs.template Ub<BF16>(K1Ub::kKInv);
        auto kres = bufs.template Ub<BF16>(K1Ub::kKRes);
        auto gtot = bufs.template Ub<float>(K1Ub::kGTotal);
        auto ident = bufs.template Ub<BF16>(K1Ub::kSmallA);

        // The identity the AIC seeds L0C with during the Neumann iteration.
        // Building it here costs one tile's worth of scalar stores and saves a
        // cross-core round trip per factor.
        AscendC::Duplicate(ident, static_cast<BF16>(0.0f), CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)1);
        for (int i = 0; i < CHUNK; ++i) {
            ident.SetValue(i * CHUNK + i, static_cast<BF16>(1.0f));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>((event_t)1);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)0);

        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kKDecayed) / 2], kdec, CHUNK * D);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kQDecayed) / 2], qdec, CHUNK * D);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kKInv) / 2], kinv, CHUNK * D);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kKRestored) / 2], kres, CHUNK * D);
        AscendC::DataCopy(wsF[Ws(span, headIdx, WorkspaceOffsets::kGTotal) / 4], gtot, D);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kIdentity) / 2], ident, CHUNK * CHUNK);
    }

    // ---------------- AIV round 2: mask, beta, (I - L) ----------------
    CATLASS_DEVICE
    void MaskAndBuild(K1AivBufs& bufs, Params const& params, int headIdx, TileSpan const& span)
    {
        auto lf = bufs.template Ub<float>(K1Ub::kLf);
        auto mf = bufs.template Ub<float>(K1Ub::kMqkF);
        // bsig occupies only CHUNK floats of kTmp; a 16x16 tile past it is free.
        auto lt = bufs.template Ub<float>(K1Ub::kTmp + 256);
        auto sa = bufs.template Ub<BF16>(K1Ub::kSmallA);
        auto sb = bufs.template Ub<BF16>(K1Ub::kSmallB);

        AscendC::GlobalTensor<float> wsF;
        AscendC::GlobalTensor<BF16> wsB;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::DataCopy(lf, wsF[Slot(span, headIdx, 0) / 4], CHUNK * CHUNK);
        AscendC::DataCopy(mf, wsF[Slot(span, headIdx, 1) / 4], CHUNK * CHUNK);

        AscendC::GlobalTensor<BF16> gmBeta;
        gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.beta));
        // beta is [T_total, H]: this chunk's values start at the first token of
        // the tile and repeat every H elements.
        const int64_t betaOff = span.tokenBase * params.H + headIdx;

        // sigmoid(beta) for all 16 rows at once. Two hardware constraints shape
        // this: the aicore has no scalar exp intrinsic, and the backend rejects
        // scalar bf16 <-> float casts outright. So beta is loaded as a vector
        // and converted with Cast on the vector unit, never element by element.
        auto bsig = bufs.template Ub<float>(K1Ub::kTmp);
        auto bwrk = bufs.template Ub<float>(K1Ub::kTmp2);
        auto braw = bufs.template Ub<BF16>(K1Ub::kScalar);

        AscendC::Duplicate(braw, static_cast<BF16>(0), CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)3);

        // blockCount rows of one bf16 each, srcStride skipping the other heads.
        // Copy the [CHUNK, H] block whole rather than gathering one element
        // per token: DataCopyPad pads each block to 32 bytes, so a strided
        // one-element-per-block copy lands the values 16 apart with garbage
        // between them.
        const int nRaw = CHUNK * params.H;
        AscendC::DataCopy(braw, gmBeta[span.tokenBase * params.H], nRaw);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        auto bwide = bufs.template Ub<float>(K1Ub::kScalar + 64);
        AscendC::Cast(bwide, braw, AscendC::RoundMode::CAST_NONE, nRaw);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)6);
        for (int r = 0; r < CHUNK; ++r) {
            bsig.SetValue(r, (r < span.actualLen)
                                 ? bwide.GetValue(r * params.H + headIdx) : 0.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)6);
        Sigmoid(bufs, bsig, bwrk, CHUNK);
        // Mask exactly as fwd_kernel1.cuh:476-492:
        //   L   : i <= j -> 0 (diagonal included); i > j -> L * sigmoid(beta[i])
        //   INV : i == j -> 1; i < j -> 0; i > j -> -L_masked
        //   Mqk : i < j -> 0 (diagonal kept)
        //
        // Built with vector ops rather than a 16x16 scalar double loop. The
        // masks are pure functions of (i, j): diff = i - j is integer-valued in
        // fp32, so clamping it to [0, 1] lands exactly on 0.0 or 1.0 and the
        // masks are exact.
        //
        // Destinations are whole tiles or row starts, which are 64 bytes apart
        // and so always 32-byte aligned. Writing from [i*CHUNK + i] to express
        // "diagonal onwards" would not be, hence the clamping.
        auto colNeg = bufs.template Ub<float>(K1Ub::kColNeg);
        auto diff = bufs.template Ub<float>(K1Ub::kDiff);
        auto mL = bufs.template Ub<float>(K1Ub::kMaskL);
        auto mLE = bufs.template Ub<float>(K1Ub::kMaskLE);

        AscendC::ArithProgression(colNeg, 0.0f, -1.0f, CHUNK);   // -j
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = 0; i < CHUNK; ++i) {
            AscendC::Adds(diff[i * CHUNK], colNeg, static_cast<float>(i), CHUNK);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Maxs(mL, diff, 0.0f, CHUNK * CHUNK);            // i > j
        AscendC::Adds(mLE, diff, 1.0f, CHUNK * CHUNK);           // i >= j
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(mL, mL, 1.0f, CHUNK * CHUNK);
        AscendC::Maxs(mLE, mLE, 0.0f, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(mLE, mLE, 1.0f, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();

        // All 16 sigmoid(beta) values in one scalar round trip, as before.
        float bs[CHUNK];
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)4);
        for (int i = 0; i < CHUNK; ++i) {
            bs[i] = (i < span.actualLen) ? bsig.GetValue(i) : 0.0f;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)4);

        // lt = L * sigmoid(beta)[row], kept strictly lower. The Neumann
        // iteration squares this, so it cannot be replaced by (I - L):
        // (I - L)^2 = I - 2L + L^2.
        for (int i = 0; i < CHUNK; ++i) {
            AscendC::Muls(lt[i * CHUNK], lf[i * CHUNK], bs[i], CHUNK);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(lt, lt, mL, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();

        // lf becomes (I - lt), with I = lowerInclusive - strictlyLower.
        AscendC::Sub(lf, mLE, mL, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(lf, lf, lt, CHUNK * CHUNK);
        // Mqk keeps the diagonal, zeroes strictly above it.
        AscendC::Mul(mf, mf, mLE, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(sa, lf, AscendC::RoundMode::CAST_RINT, CHUNK * CHUNK);
        AscendC::Cast(sb, mf, AscendC::RoundMode::CAST_RINT, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)0);

        // (I - L) into slot 0 for the AIC; Mqk is final; plain masked L into
        // slot 5, which the Neumann iteration needs in order to square it.
        // (I - L) cannot be squared in its place: (I-L)^2 = I - 2L + L^2.
        AscendC::DataCopy(wsB[Slot(span, headIdx, 0) / 2], sa, CHUNK * CHUNK);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kMqk) / 2], sb, CHUNK * CHUNK);

        // L into slot 5 for the Neumann iteration.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((event_t)0);
        AscendC::Cast(sa, lt, AscendC::RoundMode::CAST_RINT, CHUNK * CHUNK);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
        AscendC::DataCopy(wsB[Slot(span, headIdx, 5) / 2], sa, CHUNK * CHUNK);

    }

    // ---------------- AIC round 1 ----------------
    // L = k_dec @ k_inv^T and Mqk = q_dec @ k_inv^T, both [16,128]x[128,16].
    // k_inv is RowMajor [16,128] in GM; the same bytes read as ColumnMajor
    // [128,16] are k_inv^T, so the transpose is a different Nd2Nz
    // parameterization rather than any data movement. The draft set
    // ifTranspose on a zN source, for which catlass has no path.
    CATLASS_DEVICE
    void ComputeLAndMqk(K1AicBufs& bufs, Params const& params, int headIdx, TileSpan const& span)
    {
        LoadBt(bufs, params, Ws(span, headIdx, WorkspaceOffsets::kKInv));
        Gemm128(bufs, params, Ws(span, headIdx, WorkspaceOffsets::kKDecayed), Slot(span, headIdx, 0));
        Gemm128(bufs, params, Ws(span, headIdx, WorkspaceOffsets::kQDecayed), Slot(span, headIdx, 1));
    }

    // GM RowMajor [16,128] -> L1 nZ [128,16]: the transposed B operand.
    CATLASS_DEVICE
    void LoadBt(K1AicBufs& bufs, Params const& params, int64_t gmByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        auto l1B = bufs.template L1<BF16>(K1L1::kB);

        // k_inv is [CHUNK, D] RowMajor in GM. Its transpose is the same bytes
        // viewed as ColumnMajor [D, CHUNK]: element (d, t) sits at t*D + d,
        // which is exactly ColumnMajor with a column pitch of D.
        //
        // Nd2Nz describes a ColumnMajor source by its *columns*, not its rows
        // (catlass CopyGmToL1 for ColumnMajor -> nZ): nValue is the column
        // count, dValue the length of a column, srcDValue the column pitch.
        // Getting this backwards asks for 128 spans of 16 at a 128 pitch, which
        // reads out to element 16272 of a 2048-element buffer -- an out-of-
        // bounds GM read, and the aicore exception that produced.
        //
        // For nZ [D, CHUNK]: stride(1) = colsRound * C0 = 256 and stride(2) =
        // C0 = 16, so dstNzC0Stride = 256/16 = CHUNK and dstNzNStride = 1.
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = CHUNK;
        p.dValue = D;
        p.srcNdMatrixStride = 0;
        p.srcDValue = D;
        p.dstNzC0Stride = CHUNK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1B, gm[gmByte / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
    }

    // One [16,128]x[128,16] MMAD; fp32 result to GM. n=16 k=128 fits L0 easily.
    CATLASS_DEVICE
    void Gemm128(K1AicBufs& bufs, Params const& params, int64_t aByte, int64_t dstByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        AscendC::GlobalTensor<float> out;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        auto l1A = bufs.template L1<BF16>(K1L1::kA);
        auto l1B = bufs.template L1<BF16>(K1L1::kB);
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = CHUNK;
        p.dValue = D;
        p.srcNdMatrixStride = 0;
        p.srcDValue = D;
        p.dstNzC0Stride = CHUNK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A, gm[aByte / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);

        // srcStride is layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL (256 for bf16),
        // so it differs between the two operands and they cannot share params:
        //   A [16,128]  zN: stride(3) = 256  -> 1
        //   B [128,16]  nZ: stride(3) = 2048 -> 8
        AscendC::LoadData2DParams lda;
        lda.startIndex = 0;
        lda.repeatTimes = (CHUNK / 16) * (D / 16);
        lda.srcStride = (CHUNK * C0_NUM_PER_FRACTAL) / (C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL);
        lda.dstGap = 0;
        lda.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, lda);

        AscendC::LoadData2DParams ldb;
        ldb.startIndex = 0;
        ldb.repeatTimes = (D / 16) * (CHUNK / 16);
        ldb.srcStride = 1;  // reverted: srcStride=8 made a working single tile hang
        ldb.dstGap = 0;
        ldb.ifTranspose = false;
        AscendC::LoadData(l0B, l1B, ldb);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)0);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = D;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        // The MMAD reads L0A/L0B; the next GEMM's LoadData overwrites them.
        // Without M_MTE1 between the two, MTE1 can start writing L0B while the
        // cube is still reading it -- reported as "L0B read/write conflict in
        // the MTE (same address)". These helpers are called back to back with
        // the same L0 buffers, so the barrier belongs right after the Mmad.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)0);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        fp.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(out[dstByte / 4], l0C, fp);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)0);

        // Without this, only the LAST Gemm128 issued before the kernel ends
        // produces reliable GM data; every earlier one is non-deterministic.
        // Three experiments pinned it down: swapping the two calls moved the
        // corruption to whichever ran first, issuing the same call twice left
        // it corrupt, and the surviving raw fp32 output showed uninitialized
        // memory rather than wrong arithmetic. The Fixpipe write had not
        // retired before the next call's MTE2 traffic began. Gemm16 already
        // carries this barrier; Gemm128 was missing it.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);
    }

    // ---------------- AIC round 2: Neumann inverse ----------------
    // The AIV left A = (I - L) in slot 0, with L strictly lower triangular so
    // L^16 = 0 and
    //     (I + L)^{-1} = sum (-L)^k = (I - L)(I + L^2)(I + L^4)(I + L^8),
    // the decomposition the CUDA kernel uses.
    //
    // Each factor is applied as P(I + X) = P + P*X: one MMAD seeds L0C from P
    // against an identity, a second accumulates P*X with cmatrixInitVal =
    // false, and Fixpipe's F322BF16 rounds on the way out. That keeps the whole
    // iteration on the cube -- the draft tried to do the additions with vector
    // Add on the AIC, which has no vector unit.
    //
    // Squaring is sign-safe: (I - L) carries -L below the diagonal and
    // (-L)^2 = L^2, so squaring slot 0's strictly-lower part gives L^2.
    CATLASS_DEVICE
    // dst = A*B1 + A*B2, in one pass through the cube.
    //
    // Every factor of the Neumann series is P*(I + L^k), which expands to
    // P*I + P*L^k. Done as two Gemm16 calls that is two GM round trips plus a
    // third to copy the result back; done here it is one, because the L0C
    // accumulator holds the partial sum between the two Mmads and A stays
    // resident in L0A.
    void Gemm16Fused(K1AicBufs& bufs, Params const& params, int64_t aByte,
                     int64_t b1Byte, int64_t b2Byte, int64_t dstByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = bufs.template L1<BF16>(K1L1::kSmallA);
        auto l1B = bufs.template L1<BF16>(K1L1::kSmallB);
        auto l1C = bufs.template L1<BF16>(K1L1::kSmallC);
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = CHUNK;
        p.dValue = CHUNK;
        p.srcNdMatrixStride = 0;
        p.srcDValue = CHUNK;
        p.dstNzC0Stride = CHUNK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A, gm[aByte / 2], p);
        AscendC::DataCopy(l1B, gm[b1Byte / 2], p);
        AscendC::DataCopy(l1C, gm[b2Byte / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = 1;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, ld);
        AscendC::LoadData(l0B, l1B, ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        // The cube must finish reading L0B before MTE1 overwrites it with the
        // second operand -- the same "L0B read/write conflict in the MTE" that
        // Gemm16 guards against, and here the reload is immediate.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        // L0A still holds A; only B changes.
        AscendC::LoadData(l0B, l1C, ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        mp.cmatrixInitVal = false;   // accumulate onto A*B1 already in L0C
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        fp.quantPre = QuantMode_t::F322BF16;
        AscendC::GlobalTensor<BF16> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(out[dstByte / 2], l0C, fp);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)1);

        // The next call reloads L1 from the GM this Fixpipe just wrote.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
    }

    // One 16x16 gemm whose operands and result all live in L1.
    //
    // The GM-based Gemm16 writes its result out through Fixpipe and the next
    // call reads it back, which for a chain of six is six round trips and six
    // FIX_MTE2 waits to move 512 bytes at a time. Everything in the Neumann
    // chain is private to this core, so Fixpipe can target L1 directly --
    // dav_c220 implements FixpipeL0C2L1Impl -- and the next LoadData reads it
    // without memory ever being involved.
    CATLASS_DEVICE
    void Gemm16L1(K1AicBufs& bufs, uint32_t aL1, uint32_t bL1, uint32_t dstL1)
    {
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = 1;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        AscendC::LoadData(l0A, bufs.template L1<BF16>(aL1), ld);
        AscendC::LoadData(l0B, bufs.template L1<BF16>(bL1), ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        fp.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(
            bufs.template L1<BF16>(dstL1), l0C, fp);

        // The next LoadData reads what this just wrote, but from L1 rather than
        // GM, so this is FIX_MTE1 and not the FIX_MTE2 the GM version needs.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>((event_t)1);
    }

    // dst = A*B1 + A*B2, all in L1. The L1 counterpart of Gemm16Fused: each
    // Neumann factor is P*(I + L^k) = P*I + P*L^k, two Mmads accumulating into
    // one L0C with a single Fixpipe.
    CATLASS_DEVICE
    void Gemm16FusedL1(K1AicBufs& bufs, uint32_t aL1, uint32_t b1L1, uint32_t b2L1,
                       uint32_t dstL1)
    {
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = 1;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        AscendC::LoadData(l0A, bufs.template L1<BF16>(aL1), ld);
        AscendC::LoadData(l0B, bufs.template L1<BF16>(b1L1), ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        AscendC::LoadData(l0B, bufs.template L1<BF16>(b2L1), ld);   // A stays in L0A

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        mp.cmatrixInitVal = false;      // accumulate onto A*B1
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        fp.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(
            bufs.template L1<BF16>(dstL1), l0C, fp);

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>((event_t)1);
    }

    // The Neumann series with every intermediate in L1. GM is touched twice:
    // the operands in, the inverse out.
    CATLASS_DEVICE
    void ComputeNeumannL1(K1AicBufs& bufs, Params const& params, int headIdx,
                          TileSpan const& span)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = CHUNK;
        p.dValue = CHUNK;
        p.srcNdMatrixStride = 0;
        p.srcDValue = CHUNK;
        p.dstNzC0Stride = CHUNK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        // In: L, (I - L), and the identity.
        AscendC::DataCopy(bufs.template L1<BF16>(K1L1::kNeuA),
                          gm[Slot(span, headIdx, 5) / 2], p);      // L
        AscendC::DataCopy(bufs.template L1<BF16>(K1L1::kNeuPA),
                          gm[Slot(span, headIdx, 0) / 2], p);      // I - L
        AscendC::DataCopy(bufs.template L1<BF16>(K1L1::kNeuI),
                          gm[Ws(span, headIdx, WorkspaceOffsets::kIdentity) / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);

        // (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8).
        // L powers ping-pong kNeuA/kNeuB, the product kNeuPA/kNeuPB.
        Gemm16L1(bufs, K1L1::kNeuA, K1L1::kNeuA, K1L1::kNeuB);                    // L^2
        Gemm16FusedL1(bufs, K1L1::kNeuPA, K1L1::kNeuI, K1L1::kNeuB, K1L1::kNeuPB);
        Gemm16L1(bufs, K1L1::kNeuB, K1L1::kNeuB, K1L1::kNeuA);                    // L^4
        Gemm16FusedL1(bufs, K1L1::kNeuPB, K1L1::kNeuI, K1L1::kNeuA, K1L1::kNeuPA);
        Gemm16L1(bufs, K1L1::kNeuA, K1L1::kNeuA, K1L1::kNeuB);                    // L^8

        // Last factor goes to GM, where kernel2 reads it.
        Gemm16FusedFromL1ToGm(bufs, params, K1L1::kNeuPA, K1L1::kNeuI, K1L1::kNeuB,
                              Ws(span, headIdx, WorkspaceOffsets::kINV));
    }

    // Final step of the L1 chain: operands in L1, result to GM.
    CATLASS_DEVICE
    void Gemm16FusedFromL1ToGm(K1AicBufs& bufs, Params const& params, uint32_t aL1,
                               uint32_t b1L1, uint32_t b2L1, int64_t dstByte)
    {
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = 1;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        AscendC::LoadData(l0A, bufs.template L1<BF16>(aL1), ld);
        AscendC::LoadData(l0B, bufs.template L1<BF16>(b1L1), ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        AscendC::LoadData(l0B, bufs.template L1<BF16>(b2L1), ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        mp.cmatrixInitVal = false;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        fp.quantPre = QuantMode_t::F322BF16;
        AscendC::GlobalTensor<BF16> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(out[dstByte / 2], l0C, fp);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
    }

    void ComputeNeumann(K1AicBufs& bufs, Params const& params, int headIdx, TileSpan const& span)
    {
        const int64_t imL   = Slot(span, headIdx, 0);   // (I - L)
        const int64_t lbase = Slot(span, headIdx, 5);   // L itself
        const int64_t inv   = Ws(span, headIdx, WorkspaceOffsets::kINV);
        const int64_t ident = Ws(span, headIdx, WorkspaceOffsets::kIdentity);

        // (I + L)^-1 = (I - L)(I + L^2)(I + L^4)(I + L^8), with L strictly
        // lower triangular so the series terminates.
        //
        // The L powers alternate between slots 2 and 3 and the running product
        // between the unused slots 6 and 7, so nothing is ever copied back to
        // a fixed location. Squaring (I - L) instead of L would give
        // I - 2L + L^2, which is why L is kept separately in slot 5.
        const int64_t lA = Slot(span, headIdx, 2);
        const int64_t lB = Slot(span, headIdx, 3);
        const int64_t pA = Slot(span, headIdx, 6);
        const int64_t pB = Slot(span, headIdx, 7);

        Gemm16(bufs, params, lbase, lbase, lA, true, true);   // L^2
        Gemm16Fused(bufs, params, imL, ident, lA, pA);        // (I-L)(I + L^2)
        Gemm16(bufs, params, lA, lA, lB, true, true);         // L^4
        Gemm16Fused(bufs, params, pA, ident, lB, pB);         // ... (I + L^4)
        Gemm16(bufs, params, lB, lB, lA, true, true);         // L^8
        Gemm16Fused(bufs, params, pB, ident, lA, inv);        // ... (I + L^8)
    }

    // 16x16x16 MMAD over bf16 GM scratch. bf16Out selects Fixpipe's F322BF16
    // rounding so the result can feed the next MMAD without an AIV round-trip.
    CATLASS_DEVICE
    void Gemm16(K1AicBufs& bufs, Params const& params, int64_t aByte, int64_t bByte, int64_t dstByte,
                bool init, bool bf16Out)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = bufs.template L1<BF16>(K1L1::kSmallA);
        auto l1B = bufs.template L1<BF16>(K1L1::kSmallB);
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = CHUNK;
        p.dValue = CHUNK;
        p.srcNdMatrixStride = 0;
        p.srcDValue = CHUNK;
        p.dstNzC0Stride = CHUNK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A, gm[aByte / 2], p);
        AscendC::DataCopy(l1B, gm[bByte / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)2);

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = 1;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, ld);
        AscendC::LoadData(l0B, l1B, ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = CHUNK;
        mp.n = CHUNK;
        mp.k = CHUNK;
        mp.cmatrixInitVal = init;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        // The MMAD reads L0A/L0B; the next GEMM's LoadData overwrites them.
        // Without M_MTE1 between the two, MTE1 can start writing L0B while the
        // cube is still reading it -- reported as "L0B read/write conflict in
        // the MTE (same address)". These helpers are called back to back with
        // the same L0 buffers, so the barrier belongs right after the Mmad.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)4);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = CHUNK;
        fp.mSize = CHUNK;
        fp.srcStride = CHUNK;
        fp.dstStride = CHUNK;
        if (bf16Out) {
            AscendC::GlobalTensor<BF16> out;
            out.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
            fp.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(out[dstByte / 2], l0C, fp);
        } else {
            AscendC::GlobalTensor<float> out;
            out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
            fp.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(out[dstByte / 4], l0C, fp);
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)1);

        // ComputeNeumann chains nine of these, each reloading the same L1 slots
        // from the GM the previous Fixpipe just wrote. The next MTE2 must not
        // start before that Fixpipe retires, so this is a barrier -- set and
        // wait together. Previously the set had no matching wait, so nine
        // unconsumed FIX_MTE2 events accumulated and the core stalled.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
    }
};

}  // namespace flash_kda
