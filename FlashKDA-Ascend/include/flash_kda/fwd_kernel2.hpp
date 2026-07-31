#pragma once

/**
 * FlashKDA Ascend -- Kernel 2: Recurrence.
 *
 * Rewritten from scratch (see ../../STATUS.md).
 *
 * Grid: one core per (sequence, head). Each core walks that sequence's chunks
 * in order, carrying the [D, D] recurrent state. The scan is inherently serial,
 * so parallelism comes from N*H, not from the time axis.
 *
 * Per chunk, matching FlashKDA/csrc/smxx/fwd_kernel2.cuh:
 *   u       = Mqk_masked @ v  -  (INV @ k_dec) @ state       (the delta rule)
 *   out     = q_dec @ state  +  Mqk @ u
 *   state   = state * g_total  +  k_res^T @ u
 *
 * The state is held in GM, not L1. That is forced: the AIC would need it in L1
 * to feed the cube, the AIV needs it in UB to scale it by g_total, and A2 has
 * no L1<->UB path in either direction. GM is the only place both can reach.
 *
 * GEMMs run at full tile width -- [16,128]@[128,128] is one Mmad, not 64 of
 * 16x16x16. L0A needs 4 KB, L0B 32 KB, L0C 8 KB against 64/64/128 KB.
 *
 * State orientation is [key, value] internally, and the host transposes at the
 * GM boundary to match the reference layout.
 */

#include "flash_kda/layout.hpp"
#include "flash_kda/utils.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include <type_traits>

#include "kernel_operator.h"

namespace flash_kda {

constexpr Catlass::Arch::FlagID K2_ELEM_READY = 3;
constexpr Catlass::Arch::FlagID K2_MMA_READY = 4;

// UB map, AIV only.
struct K2Ub {
    static constexpr uint32_t kV      = 0;                      // [16,128] bf16
    static constexpr uint32_t kU      = kV + CHUNK * D * 2;      // [16,128] bf16
    static constexpr uint32_t kF32A   = kU + CHUNK * D * 2;      // [16,128] f32
    static constexpr uint32_t kF32B   = kF32A + CHUNK * D * 4;   // [16,128] f32
    static constexpr uint32_t kGTotal = kF32B + CHUNK * D * 4;   // [128] f32
    static constexpr uint32_t kBeta   = kGTotal + D * 4;         // [16] f32
    static constexpr uint32_t kStateA = kBeta + 64;              // [128,128] f32 row block
    static constexpr uint32_t kStateB = kStateA + D * D * 4;
    static constexpr uint32_t kScalar = kStateB + D * D * 4;
    static constexpr uint32_t kEnd    = kScalar + 256;
};
static_assert(K2Ub::kEnd < ArchTag::UB_SIZE, "kernel2 UB budget exceeded");

// L1 map, AIC only. Two [16,128] operands plus the [128,128] state.
struct K2L1 {
    static constexpr uint32_t kA     = 0;
    static constexpr uint32_t kB     = kA + CHUNK * D * 2;
    static constexpr uint32_t kState = kB + CHUNK * D * 2;
    static constexpr uint32_t kSmall = kState + D * D * 2;
    static constexpr uint32_t kEnd   = kSmall + 512;
};
static_assert(K2L1::kEnd < ArchTag::L1_SIZE, "kernel2 L1 budget exceeded");

class FwdRecurrenceKernel {
public:
    using Params = FwdParams;

    CATLASS_DEVICE
    FwdRecurrenceKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        if (static_cast<int>(coreIdx) >= params.N * params.H) {
            return;
        }
        const int headIdx = static_cast<int>(coreIdx) % params.H;
        const int seqIdx = static_cast<int>(coreIdx) / params.H;

        int64_t bos = 0;
        int seqLen = 0;
        int tileBase = 0;
        ResolveSeq(params, seqIdx, bos, seqLen, tileBase);
        const int nTiles = (seqLen + CHUNK - 1) / CHUNK;
        const bool active = (subIdx == 0);

        if (active) {
            InitState(params, seqIdx, headIdx);
        }

        // Every chunk performs exactly two handshakes in both directions,
        // whether or not this AIV is the active sub-core, so the AIC never
        // waits on a flag that will not arrive.
        for (int t = 0; t < nTiles; ++t) {
            const int tileIdx = tileBase + t;
            int len = seqLen - t * CHUNK;
            if (len > CHUNK) {
                len = CHUNK;
            }

            // Round 1: build u from v and the delta-rule correction.
            if (active) {
                BuildU(params, headIdx, tileIdx, bos + t * CHUNK, len);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if (subIdx == 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
                Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
            }

            // Round 2: combine out, then decay and update the state.
            if (active) {
                FinishOut(params, headIdx, tileIdx, bos + t * CHUNK, len);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if (subIdx == 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
                Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
            }

            // Round 2's GEMMs are done: out can be summed and stored, and the
            // state advanced with k_res^T @ u.
            if (active) {
                StoreOut(params, headIdx, tileIdx, bos + t * CHUNK, len);
                DecayState(params, seqIdx, headIdx, tileIdx);
            }
        }

        if (active && params.has_state_out != 0) {
            StoreFinalState(params, seqIdx, headIdx);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        if (static_cast<int>(coreIdx) >= params.N * params.H) {
            return;
        }
        const int headIdx = static_cast<int>(coreIdx) % params.H;
        const int seqIdx = static_cast<int>(coreIdx) / params.H;

        int64_t bos = 0;
        int seqLen = 0;
        int tileBase = 0;
        ResolveSeq(params, seqIdx, bos, seqLen, tileBase);
        const int nTiles = (seqLen + CHUNK - 1) / CHUNK;

        for (int t = 0; t < nTiles; ++t) {
            const int tileIdx = tileBase + t;

            // Round 1: the two GEMMs u depends on.
            Catlass::Arch::CrossCoreWaitFlag(elemReady_);
            PreGemms(params, seqIdx, headIdx, tileIdx);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);

            // Round 2: out contribution, then the state's rank-16 update.
            Catlass::Arch::CrossCoreWaitFlag(elemReady_);
            PostGemms(params, seqIdx, headIdx, tileIdx);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);
        }
    }

private:
    CATLASS_DEVICE
    void ResolveSeq(Params const& params, int seqIdx, int64_t& bos, int& seqLen, int& tileBase)
    {
        if (params.is_varlen == 0) {
            seqLen = params.T_total / params.N;
            bos = static_cast<int64_t>(seqIdx) * seqLen;
            tileBase = seqIdx * ((seqLen + CHUNK - 1) / CHUNK);
            return;
        }
        AscendC::GlobalTensor<int64_t> gmCu;
        gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens));
        int consumed = 0;
        for (int s = 0; s < seqIdx; ++s) {
            const int len = static_cast<int>(gmCu.GetValue(s + 1) - gmCu.GetValue(s));
            consumed += (len + CHUNK - 1) / CHUNK;
        }
        bos = gmCu.GetValue(seqIdx);
        seqLen = static_cast<int>(gmCu.GetValue(seqIdx + 1) - bos);
        tileBase = consumed;
    }

    CATLASS_DEVICE
    int64_t Ws(Params const& params, int tileIdx, int headIdx, uint32_t field) const
    {
        return static_cast<int64_t>(tileIdx) * params.H * WorkspaceSizes::kPerTile +
               static_cast<int64_t>(headIdx) * WorkspaceSizes::kPerTile + field;
    }

    CATLASS_DEVICE
    int64_t Slot(Params const& params, int tileIdx, int headIdx, int i) const
    {
        return Ws(params, tileIdx, headIdx, WorkspaceOffsets::kScratch) +
               static_cast<int64_t>(i) * WorkspaceSizes::kScratchSlot;
    }

    // The live state for this (sequence, head), in GM.
    CATLASS_DEVICE
    int64_t StateOff(Params const& params, int seqIdx, int headIdx) const
    {
        return params.state_ws_offset +
               (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D * 4;
    }

    // ---------------- AIV ----------------
    // ---- state layout at the API boundary ----
    //
    // GM initial_state / final_state are [value, key]; internally the state is
    // [key, value] because that is the orientation every GEMM here wants. These
    // two helpers do the transpose, and only run once per (sequence, head).
    //
    // DataCopyPad gathers a strided column: D blocks of one element with a
    // (D-1)-element gap, which is destination row r drawn from gm[:, r].

    template <class T>
    CATLASS_DEVICE void StateGmToUbT(AscendC::LocalTensor<float> dst,
                                     AscendC::GlobalTensor<T> gm, int64_t base)
    {
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(D);
        p.blockLen = static_cast<uint32_t>(sizeof(T));
        p.srcStride = static_cast<uint32_t>((D - 1) * sizeof(T));
        p.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};

        auto staging = resource_.ubBuf.template GetBufferByByte<T>(K2Ub::kStateB);
        for (int r = 0; r < D; ++r) {
            AscendC::DataCopyPad(staging[r * D], gm[base + r], p, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)6);
        if constexpr (std::is_same_v<T, float>) {
            // Already fp32: the staging buffer is the result. Cast<float,float>
            // is not a valid conversion.
            AscendC::DataCopy(dst, staging.template ReinterpretCast<float>(), D * D);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(dst, staging, AscendC::RoundMode::CAST_NONE, D * D);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <class T>
    CATLASS_DEVICE void StateUbToGmT(AscendC::GlobalTensor<T> gm, int64_t base,
                                     AscendC::LocalTensor<T> src)
    {
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(D);
        p.blockLen = static_cast<uint32_t>(sizeof(T));
        p.srcStride = 0;
        p.dstStride = static_cast<uint32_t>((D - 1) * sizeof(T));
        for (int r = 0; r < D; ++r) {
            AscendC::DataCopyPad(gm[base + r], src[r * D], p);
        }
    }

    CATLASS_DEVICE
    void InitState(Params const& params, int seqIdx, int headIdx)
    {
        auto sa = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kStateA);
        AscendC::GlobalTensor<float> gmS;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        const int64_t dst = StateOff(params, seqIdx, headIdx);

        if (params.has_state_in != 0) {
            // initial_state is [N, H, D, D]; fp32 or bf16 per has_state_fp32.
            if (params.state_fp32 != 0) {
                AscendC::GlobalTensor<float> gmIn;
                gmIn.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.initial_state));
                const int64_t src = (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D;
                // GM is [value, key]; sa is [key, value].
                StateGmToUbT<float>(sa, gmIn, src);
            } else {
                AscendC::GlobalTensor<BF16> gmIn;
                gmIn.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.initial_state));
                const int64_t src = (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D;
                // GM is [value, key]; sa is [key, value].
                StateGmToUbT<BF16>(sa, gmIn, src);
            }
        } else {
            AscendC::Duplicate(sa, 0.0f, D * D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::DataCopy(gmS[dst / 4], sa, D * D);
    }

    // u = Mqk @ v - (INV @ k_dec) @ state, staged so the AIC can do both GEMMs
    // in one round. Here we only prepare v (and zero its tail rows) and the
    // bf16 copy of the state the cube will read.
    CATLASS_DEVICE
    void BuildU(Params const& params, int headIdx, int tileIdx, int64_t tokenBase, int len)
    {
        auto vb = resource_.ubBuf.template GetBufferByByte<BF16>(K2Ub::kV);
        AscendC::GlobalTensor<BF16> gmV;
        gmV.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.v));

        // Tail rows must be zeroed, not left as whatever GM held: kernel 1
        // zeroes k's tail so k_res's tail columns vanish, but v is read here
        // and a garbage row would poison the state through k_res^T @ u.
        AscendC::Duplicate(vb, static_cast<BF16>(0.0f), CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)0);
        for (int r = 0; r < len; ++r) {
            const int64_t off = (tokenBase + r) * params.H * D + static_cast<int64_t>(headIdx) * D;
            AscendC::DataCopy(vb[r * D], gmV[off], D);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>((event_t)0);

        // v into scratch slot 2 for the cube.
        AscendC::GlobalTensor<BF16> wsB;
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 2) / 2], vb, CHUNK * D);

        StateToBf16(params, tileIdx, headIdx);
    }

    // The cube reads the state as bf16; the live copy is fp32 in GM.
    CATLASS_DEVICE
    void StateToBf16(Params const& params, int tileIdx, int headIdx)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const int seqIdx = static_cast<int>(coreIdx) / params.H;

        auto sf = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kStateA);
        auto sb = resource_.ubBuf.template GetBufferByByte<BF16>(K2Ub::kStateB);
        AscendC::GlobalTensor<float> gmS;
        AscendC::GlobalTensor<BF16> wsB;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::DataCopy(sf, gmS[StateOff(params, seqIdx, headIdx) / 4], D * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)1);
        AscendC::Cast(sb, sf, AscendC::RoundMode::CAST_RINT, D * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 3) / 2], sb, D * D);
    }

    // After round 1 the cube left  slot4 = Mqk @ v  and  slot5 = (INV@k_dec)@state.
    // u = slot4 - slot5, then out = slot6 (q_dec@state) + Mqk @ u needs another
    // GEMM, so u goes back out for round 2.
    CATLASS_DEVICE
    void FinishOut(Params const& params, int headIdx, int tileIdx, int64_t tokenBase, int len)
    {
        auto a = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kF32A);
        auto b = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kF32B);
        auto ub = resource_.ubBuf.template GetBufferByByte<BF16>(K2Ub::kU);

        AscendC::GlobalTensor<float> wsF;
        AscendC::GlobalTensor<BF16> wsB;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::DataCopy(a, wsF[Slot(params, tileIdx, headIdx, 4) / 4], CHUNK * D);
        AscendC::DataCopy(b, wsF[Slot(params, tileIdx, headIdx, 5) / 4], CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)2);

        AscendC::Sub(a, a, b, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ub, a, AscendC::RoundMode::CAST_RINT, CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)2);

        // u for round 2's two GEMMs (Mqk @ u and k_res^T @ u).
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 2) / 2], ub, CHUNK * D);
        (void)tokenBase;
        (void)len;
    }

    // state = state * g_total + (k_res^T @ u), the latter left in slot 7 by the
    // cube in round 2 of the *previous* handshake half.
    CATLASS_DEVICE
    void DecayState(Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        auto sf = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kStateA);
        auto upd = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kStateB);
        auto gt = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kGTotal);

        AscendC::GlobalTensor<float> gmS;
        AscendC::GlobalTensor<float> wsF;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        const int64_t stateOff = StateOff(params, seqIdx, headIdx);
        AscendC::DataCopy(sf, gmS[stateOff / 4], D * D);
        AscendC::DataCopy(upd, wsF[Slot(params, tileIdx, headIdx, 7) / 4], D * D);
        // g_total is already exponentiated by kernel 1; exponentiating it again
        // here was one of the draft's algorithmic errors.
        AscendC::DataCopy(gt, wsF[Ws(params, tileIdx, headIdx, WorkspaceOffsets::kGTotal) / 4], D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)3);

        // State is [key, value]; the decay is per key row, so g_total broadcasts
        // down rows one at a time.
        for (int r = 0; r < D; ++r) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)0);
            const float g = gt.GetValue(r);
            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)0);
            AscendC::Muls(sf[r * D], sf[r * D], g, D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(sf, sf, upd, D * D);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)3);
        AscendC::DataCopy(gmS[stateOff / 4], sf, D * D);
    }

    // out = slot6 + slot8, both fp32 from the cube; rounded once to bf16 as the
    // reference does, and only real rows are written.
    CATLASS_DEVICE
    void StoreOut(Params const& params, int headIdx, int tileIdx, int64_t tokenBase, int len)
    {
        auto a = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kF32A);
        auto b = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kF32B);
        auto ob = resource_.ubBuf.template GetBufferByByte<BF16>(K2Ub::kU);

        AscendC::GlobalTensor<float> wsF;
        AscendC::GlobalTensor<BF16> gmOut;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.out));

        AscendC::DataCopy(a, wsF[Slot(params, tileIdx, headIdx, 6) / 4], CHUNK * D);
        AscendC::DataCopy(b, wsF[Slot(params, tileIdx, headIdx, 8) / 4], CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)4);

        AscendC::Add(a, a, b, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ob, a, AscendC::RoundMode::CAST_RINT, CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)4);

        for (int r = 0; r < len; ++r) {
            const int64_t off = (tokenBase + r) * params.H * D + static_cast<int64_t>(headIdx) * D;
            AscendC::DataCopy(gmOut[off], ob[r * D], D);
        }
    }

    CATLASS_DEVICE
    void StoreFinalState(Params const& params, int seqIdx, int headIdx)
    {
        auto sf = resource_.ubBuf.template GetBufferByByte<float>(K2Ub::kStateA);
        AscendC::GlobalTensor<float> gmS;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        AscendC::DataCopy(sf, gmS[StateOff(params, seqIdx, headIdx) / 4], D * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)5);

        const int64_t dst = (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D;
        if (params.state_fp32 != 0) {
            AscendC::GlobalTensor<float> gmFs;
            gmFs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.final_state));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            // sf is [key, value]; GM wants [value, key].
            StateUbToGmT<float>(gmFs, dst, sf);
        } else {
            auto sb = resource_.ubBuf.template GetBufferByByte<BF16>(K2Ub::kStateB);
            AscendC::Cast(sb, sf, AscendC::RoundMode::CAST_RINT, D * D);
            AscendC::GlobalTensor<BF16> gmFs;
            gmFs.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.final_state));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            // sb is [key, value]; GM wants [value, key].
            StateUbToGmT<BF16>(gmFs, dst, sb);
        }
    }

    // ---------------- AIC ----------------
    // Round 1: slot4 = Mqk @ v, slot5 = (INV @ k_dec) @ state, slot6 = q_dec @ state.
    CATLASS_DEVICE
    void PreGemms(Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        const int64_t mqk = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kMqk);
        const int64_t inv = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kINV);
        const int64_t kdec = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kKDecayed);
        const int64_t qdec = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kQDecayed);
        const int64_t v = Slot(params, tileIdx, headIdx, 2);
        const int64_t state = Slot(params, tileIdx, headIdx, 3);

        // Mqk[16,16] @ v[16,128]
        Gemm(params, mqk, v, Slot(params, tileIdx, headIdx, 4), CHUNK, D, CHUNK, false);
        // INV[16,16] @ k_dec[16,128] -> slot0 as bf16, then @ state[128,128]
        Gemm(params, inv, kdec, Slot(params, tileIdx, headIdx, 0), CHUNK, D, CHUNK, true);
        Gemm(params, Slot(params, tileIdx, headIdx, 0), state,
             Slot(params, tileIdx, headIdx, 5), CHUNK, D, D, false);
        // q_dec[16,128] @ state[128,128]
        Gemm(params, qdec, state, Slot(params, tileIdx, headIdx, 6), CHUNK, D, D, false);
        (void)seqIdx;
    }

    // Round 2: slot8 = Mqk @ u, slot7 = k_res^T @ u.
    CATLASS_DEVICE
    void PostGemms(Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        const int64_t mqk = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kMqk);
        const int64_t kres = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kKRestored);
        const int64_t u = Slot(params, tileIdx, headIdx, 2);

        Gemm(params, mqk, u, Slot(params, tileIdx, headIdx, 8), CHUNK, D, CHUNK, false);
        // k_res^T[128,16] @ u[16,128]: the transpose comes from reading k_res's
        // RowMajor bytes as ColumnMajor, not from any data movement.
        GemmAt(params, kres, u, Slot(params, tileIdx, headIdx, 7));
        (void)seqIdx;
    }

    // C[m,n] = A[m,k] @ B[k,n], all operands bf16 in GM scratch.
    CATLASS_DEVICE
    void Gemm(Params const& params, int64_t aByte, int64_t bByte, int64_t cByte,
              int m, int n, int k, bool bf16Out)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = resource_.l1Buf.template GetBufferByByte<BF16>(K2L1::kA);
        auto l1B = resource_.l1Buf.template GetBufferByByte<BF16>(K2L1::kState);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<BF16>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<BF16>(0);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);

        LoadNd2Nz(l1A, gm, aByte, m, k);
        LoadNd2Nz(l1B, gm, bByte, k, n);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        ld.repeatTimes = (m / 16) * (k / 16);
        AscendC::LoadData(l0A, l1A, ld);
        ld.repeatTimes = (k / 16) * (n / 16);
        AscendC::LoadData(l0B, l1B, ld);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)0);

        AscendC::MmadParams mp;
        mp.m = m;
        mp.n = n;
        mp.k = k;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)0);

        Drain(params, l0C, cByte, m, n, bf16Out);
    }

    // C[128,128] = A^T[128,16] @ B[16,128], A stored RowMajor [16,128].
    CATLASS_DEVICE
    void GemmAt(Params const& params, int64_t aByte, int64_t bByte, int64_t cByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = resource_.l1Buf.template GetBufferByByte<BF16>(K2L1::kA);
        auto l1B = resource_.l1Buf.template GetBufferByByte<BF16>(K2L1::kB);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<BF16>(0);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<BF16>(0);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(0);

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);

        // ColumnMajor view of the same bytes: nZ in L1, which is the layout
        // LoadData's transposing A path expects.
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = D;
        p.dValue = CHUNK;
        p.srcNdMatrixStride = 0;
        p.srcDValue = D;
        p.dstNzC0Stride = D;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A, gm[aByte / 2], p);
        LoadNd2Nz(l1B, gm, bByte, CHUNK, D);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);

        AscendC::LoadData2DParams la;
        la.startIndex = 0;
        la.srcStride = 1;
        la.dstGap = 0;
        la.ifTranspose = true;
        la.repeatTimes = (D / 16) * (CHUNK / 16);
        AscendC::LoadData(l0A, l1A, la);

        AscendC::LoadData2DParams lb;
        lb.startIndex = 0;
        lb.srcStride = 1;
        lb.dstGap = 0;
        lb.ifTranspose = false;
        lb.repeatTimes = (CHUNK / 16) * (D / 16);
        AscendC::LoadData(l0B, l1B, lb);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = D;
        mp.n = D;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        Drain(params, l0C, cByte, D, D, false);
    }

    CATLASS_DEVICE
    void LoadNd2Nz(AscendC::LocalTensor<BF16> dst, AscendC::GlobalTensor<BF16> src,
                   int64_t byteOff, int rows, int cols)
    {
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = rows;
        p.dValue = cols;
        p.srcNdMatrixStride = 0;
        p.srcDValue = cols;
        p.dstNzC0Stride = rows;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst, src[byteOff / 2], p);
    }

    CATLASS_DEVICE
    void Drain(Params const& params, AscendC::LocalTensor<float> l0C, int64_t cByte,
               int m, int n, bool bf16Out)
    {
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = n;
        fp.mSize = m;
        fp.srcStride = m;
        fp.dstStride = n;
        if (bf16Out) {
            AscendC::GlobalTensor<BF16> out;
            out.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
            fp.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<BF16, float, AscendC::CFG_ROW_MAJOR>(out[cByte / 2], l0C, fp);
        } else {
            AscendC::GlobalTensor<float> out;
            out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
            fp.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(out[cByte / 4], l0C, fp);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((event_t)0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);
    }

    Catlass::Arch::Resource<ArchTag> resource_;
    Catlass::Arch::CrossCoreFlag elemReady_{K2_ELEM_READY};
    Catlass::Arch::CrossCoreFlag mmaReady_{K2_MMA_READY};
};

}  // namespace flash_kda
