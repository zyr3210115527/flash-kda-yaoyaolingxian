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
 * Two handshakes per tile:
 *   AIV: normalize, gate, cumsum, decay, store   -> elemReady
 *   AIC: L = k_dec @ k_inv^T, Mqk = q_dec @ k_inv^T -> mmaReady
 *   AIV: tril mask + beta sigmoid, build (I - L)  -> elemReady
 *   AIC: Neumann inverse, writes INV             -> mmaReady
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
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"

namespace flash_kda {

// catlass reserves flag ids 8/9/10 and caps user flags at 7.
constexpr Catlass::Arch::FlagID K1_ELEM_READY = 1;
constexpr Catlass::Arch::FlagID K1_MMA_READY = 2;

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
    static constexpr uint32_t kScalar = kReduce + 1024;
    static constexpr uint32_t kEnd    = kScalar + 256;
};
static_assert(K1Ub::kEnd < ArchTag::UB_SIZE, "kernel1 UB budget exceeded");

// L1 map, AIC only.
struct K1L1 {
    static constexpr uint32_t kA      = 0;
    static constexpr uint32_t kB      = kA + CHUNK * D * 2;
    static constexpr uint32_t kSmallA = kB + CHUNK * D * 2;
    static constexpr uint32_t kSmallB = kSmallA + 512;
    static constexpr uint32_t kEnd    = kSmallB + 512;
};
static_assert(K1L1::kEnd < ArchTag::L1_SIZE, "kernel1 L1 budget exceeded");

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

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        // GetBlockIdx() on the AIV is the sub-core index; dividing by
        // GetSubBlockNum() recovers the index its paired AIC sees. Without this
        // the two AIVs process different tiles than the AIC they sync with.
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        if (static_cast<int>(coreIdx) >= params.total_tiles * params.H) {
            return;
        }
        const int headIdx = static_cast<int>(coreIdx) % params.H;
        const int tileIdx = static_cast<int>(coreIdx) / params.H;

        TileSpan span;
        ResolveTile(params, tileIdx, span);

        // An out-of-range tile must still complete both handshakes or its AIC
        // blocks forever. The draft returned early and deadlocked on varlen.
        const bool active = span.valid && (subIdx == 0);

        if (active) {
            Prepare(params, headIdx, span);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // Only the subcore that did the work signals. Both subcores signalling
        // sets the flag twice per round against a single AIC wait, and the
        // leftover counts hang teardown after the kernel itself has finished.
        if (subIdx == 0) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
            Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
        }
        if (active) {
            MaskAndBuild(params, headIdx, span);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if (subIdx == 0) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
            Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        if (static_cast<int>(coreIdx) >= params.total_tiles * params.H) {
            return;
        }
        const int headIdx = static_cast<int>(coreIdx) % params.H;
        const int tileIdx = static_cast<int>(coreIdx) / params.H;

        TileSpan span;
        ResolveTile(params, tileIdx, span);

        Catlass::Arch::CrossCoreWaitFlag(elemReady_);
        if (span.valid) {
            ComputeLAndMqk(params, headIdx, span);
        }
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);

        Catlass::Arch::CrossCoreWaitFlag(elemReady_);
        if (span.valid) {
            ComputeNeumann(params, headIdx, span);
        }
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);
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
               static_cast<int64_t>(i) * WorkspaceSizes::kScratchSlot;
    }

    // ---------------- AIV round 1 ----------------
    CATLASS_DEVICE
    void Prepare(Params const& params, int headIdx, TileSpan const& span)
    {
        auto qb = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kQ);
        auto kb = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kK);
        auto gb = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kG);
        auto qf = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kQf);
        auto kf = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kKf);
        auto gc = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kGc);
        auto tmp = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kTmp);
        auto tmp2 = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kTmp2);
        auto gtot = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kGTotal);
        auto dtb = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kDtBias);

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
        for (int r = 0; r < span.actualLen; ++r) {
            NormalizeRow(qf, tmp, r);
            NormalizeRow(kf, tmp, r);
        }

        AscendC::GlobalTensor<float> gmALog;
        gmALog.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.A_log));
        const float aExp = ExpViaVector(gmALog.GetValue(headIdx));

        // Gate activation, then an inclusive cumsum down the rows. Rows past
        // the tail contribute nothing, which is what lets kernel 2 skip tail
        // masking entirely.
        AscendC::Duplicate(gtot, 0.0f, D);
        AscendC::PipeBarrier<PIPE_V>();
        for (int r = 0; r < CHUNK; ++r) {
            if (r < span.actualLen) {
                AscendC::Cast(tmp, gb[r * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tmp, tmp, dtb, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(tmp, tmp, aExp, D);
                AscendC::PipeBarrier<PIPE_V>();
                Sigmoid(tmp, tmp2, D);
                AscendC::Muls(tmp, tmp, params.gate_scale, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(gtot, gtot, tmp, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Adds(gc[r * D], gtot, 0.0f, D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // g_total is stored already exponentiated, as in CUDA; kernel 2 must
        // not exponentiate it a second time.
        AscendC::Exp(gtot, gtot, D);
        AscendC::PipeBarrier<PIPE_V>();

        Decay(params, span, qf, kf, gc, tmp, gtot);
        Store(params, headIdx, span);
    }

    // The aicore has no scalar exp, so a one-off exponential has to be staged
    // through UB and run on the vector unit. One datablock (8 floats) is the
    // minimum useful width.
    CATLASS_DEVICE
    float ExpViaVector(float x)
    {
        auto pad = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kScalar);
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
    void NormalizeRow(AscendC::LocalTensor<float> mat, AscendC::LocalTensor<float> tmp, int r)
    {
        auto work = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kReduce);
        auto sc = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kScalar);

        AscendC::Mul(tmp, mat[r * D], mat[r * D], D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(sc, tmp, work, D);
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)0);
        const float inv = 1.0f / sqrt(sc.GetValue(0) + 1e-6f);
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)1);
        AscendC::Muls(mat[r * D], mat[r * D], inv, D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // sigmoid(x) = 1/(1+exp(-x)), using scratch to avoid an in-place Div.
    CATLASS_DEVICE
    void Sigmoid(AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> scratch, int n)
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
    void Decay(Params const& params, TileSpan const& span,
               AscendC::LocalTensor<float> qf, AscendC::LocalTensor<float> kf,
               AscendC::LocalTensor<float> gc, AscendC::LocalTensor<float> tmp,
               AscendC::LocalTensor<float> gtot)
    {
        auto kdec = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKDec);
        auto qdec = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kQDec);
        auto kinv = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKInv);
        auto kres = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKRes);

        AscendC::Duplicate(kdec, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(qdec, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(kinv, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::Duplicate(kres, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();

        for (int r = 0; r < span.actualLen; ++r) {
            const int off = r * D;

            AscendC::Exp(tmp, gc[off], D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp, tmp, kf[off], D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kdec[off], tmp, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(tmp, gc[off], D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp, tmp, qf[off], D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmp, tmp, params.scale, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qdec[off], tmp, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(tmp, gc[off], -1.0f, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmp, tmp, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp, tmp, kf[off], D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kinv[off], tmp, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(tmp, tmp, gtot, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kres[off], tmp, AscendC::RoundMode::CAST_RINT, D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    CATLASS_DEVICE
    void Store(Params const& params, int headIdx, TileSpan const& span)
    {
        AscendC::GlobalTensor<BF16> wsB;
        AscendC::GlobalTensor<float> wsF;
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        auto kdec = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKDec);
        auto qdec = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kQDec);
        auto kinv = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKInv);
        auto kres = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kKRes);
        auto gtot = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kGTotal);
        auto ident = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kSmallA);

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
    void MaskAndBuild(Params const& params, int headIdx, TileSpan const& span)
    {
        auto lf = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kLf);
        auto mf = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kMqkF);
        auto sa = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kSmallA);
        auto sb = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kSmallB);

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
        auto bsig = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kTmp);
        auto bwrk = resource_.ubBuf.template GetBufferByByte<float>(K1Ub::kTmp2);
        auto braw = resource_.ubBuf.template GetBufferByByte<BF16>(K1Ub::kScalar);

        AscendC::Duplicate(braw, static_cast<BF16>(0), CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)3);

        // blockCount rows of one bf16 each, srcStride skipping the other heads.
        AscendC::DataCopyExtParams bp{
            static_cast<uint16_t>(span.actualLen),
            static_cast<uint32_t>(sizeof(BF16)),
            static_cast<uint32_t>((params.H - 1) * sizeof(BF16)),
            0, 0};
        AscendC::DataCopyPadExtParams<BF16> bpad{false, 0, 0, static_cast<BF16>(0)};
        AscendC::DataCopyPad(braw, gmBeta[betaOff], bp, bpad);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::Cast(bsig, braw, AscendC::RoundMode::CAST_NONE, CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        Sigmoid(bsig, bwrk, CHUNK);
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)4);

        // Mask exactly as fwd_kernel1.cuh:476-492:
        //   L   : i <= j -> 0 (diagonal included); i > j -> L * sigmoid(beta[i])
        //   INV : i == j -> 1; i < j -> 0; i > j -> -L_masked
        //   Mqk : i < j -> 0 (diagonal kept)
        // Masks are built in fp32 scalars, in place, then converted to bf16 in
        // one vector Cast each -- scalar bf16 casts are not representable.
        for (int i = 0; i < CHUNK; ++i) {
            const float bs = (i < span.actualLen) ? bsig.GetValue(i) : 0.0f;
            for (int j = 0; j < CHUNK; ++j) {
                float v;
                if (i == j) {
                    v = 1.0f;
                } else if (i < j) {
                    v = 0.0f;
                } else {
                    v = -(lf.GetValue(i * CHUNK + j) * bs);
                }
                lf.SetValue(i * CHUNK + j, v);
                if (i < j) {
                    mf.SetValue(i * CHUNK + j, 0.0f);
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)4);
        AscendC::Cast(sa, lf, AscendC::RoundMode::CAST_RINT, CHUNK * CHUNK);
        AscendC::Cast(sb, mf, AscendC::RoundMode::CAST_RINT, CHUNK * CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)0);

        // (I - L) into slot 0 for the AIC; Mqk is final.
        AscendC::DataCopy(wsB[Slot(span, headIdx, 0) / 2], sa, CHUNK * CHUNK);
        AscendC::DataCopy(wsB[Ws(span, headIdx, WorkspaceOffsets::kMqk) / 2], sb, CHUNK * CHUNK);
    }

    // ---------------- AIC round 1 ----------------
    // L = k_dec @ k_inv^T and Mqk = q_dec @ k_inv^T, both [16,128]x[128,16].
    // k_inv is RowMajor [16,128] in GM; the same bytes read as ColumnMajor
    // [128,16] are k_inv^T, so the transpose is a different Nd2Nz
    // parameterization rather than any data movement. The draft set
    // ifTranspose on a zN source, for which catlass has no path.
    CATLASS_DEVICE
    void ComputeLAndMqk(Params const& params, int headIdx, TileSpan const& span)
    {
        LoadBt(params, Ws(span, headIdx, WorkspaceOffsets::kKInv));
        Gemm128(params, Ws(span, headIdx, WorkspaceOffsets::kKDecayed), Slot(span, headIdx, 0));
        Gemm128(params, Ws(span, headIdx, WorkspaceOffsets::kQDecayed), Slot(span, headIdx, 1));
    }

    // GM RowMajor [16,128] -> L1 nZ [128,16]: the transposed B operand.
    CATLASS_DEVICE
    void LoadBt(Params const& params, int64_t gmByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        auto l1B = resource_.l1Buf.template GetBufferByByte<BF16>(K1L1::kB);

        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = D;
        p.dValue = CHUNK;
        p.srcNdMatrixStride = 0;
        p.srcDValue = D;
        p.dstNzC0Stride = D;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1B, gm[gmByte / 2], p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
    }

    // One [16,128]x[128,16] MMAD; fp32 result to GM. n=16 k=128 fits L0 easily.
    CATLASS_DEVICE
    void Gemm128(Params const& params, int64_t aByte, int64_t dstByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        AscendC::GlobalTensor<float> out;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        auto l1A = resource_.l1Buf.template GetBufferByByte<BF16>(K1L1::kA);
        auto l1B = resource_.l1Buf.template GetBufferByByte<BF16>(K1L1::kB);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<BF16>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<BF16>(0);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

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
    void ComputeNeumann(Params const& params, int headIdx, TileSpan const& span)
    {
        const int64_t a = Slot(span, headIdx, 0);   // (I - L)
        const int64_t l = Slot(span, headIdx, 2);   // L^(2^j)
        const int64_t t = Slot(span, headIdx, 3);   // temp
        const int64_t p = Slot(span, headIdx, 4);   // running product
        const int64_t ident = Ws(span, headIdx, WorkspaceOffsets::kIdentity);

        Gemm16(params, a, a, l, true, true);        // L^2
        Gemm16(params, a, ident, p, true, true);    // P <- (I - L)

        for (int j = 0; j < 3; ++j) {
            Gemm16(params, p, ident, t, true, true);   // T <- P
            Gemm16(params, p, l, t, false, true);      // T <- P + P*L^(2^(j+1))
            Gemm16(params, t, ident, p, true, true);   // P <- T
            if (j < 2) {
                Gemm16(params, l, l, t, true, true);   // L <- L^2
                Gemm16(params, t, ident, l, true, true);
            }
        }

        // P is the inverse; publish it.
        Gemm16(params, p, ident, Ws(span, headIdx, WorkspaceOffsets::kINV), true, true);
    }

    // 16x16x16 MMAD over bf16 GM scratch. bf16Out selects Fixpipe's F322BF16
    // rounding so the result can feed the next MMAD without an AIV round-trip.
    CATLASS_DEVICE
    void Gemm16(Params const& params, int64_t aByte, int64_t bByte, int64_t dstByte,
                bool init, bool bf16Out)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = resource_.l1Buf.template GetBufferByByte<BF16>(K1L1::kSmallA);
        auto l1B = resource_.l1Buf.template GetBufferByByte<BF16>(K1L1::kSmallB);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<BF16>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<BF16>(0);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        if (init) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        }

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
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
    }

    Catlass::Arch::Resource<ArchTag> resource_;
    Catlass::Arch::CrossCoreFlag elemReady_{K1_ELEM_READY};
    Catlass::Arch::CrossCoreFlag mmaReady_{K1_MMA_READY};
};

}  // namespace flash_kda
