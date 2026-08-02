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
// RunFusedAll hands off between AIC and AIV with cross-core flags.
#include "catlass/arch/cross_core_sync.hpp"
#include <type_traits>

#include "kernel_operator.h"

namespace flash_kda {


// UB map, AIV only.
// K2_BETA_SLOT: scratch slot 1 carries sigmoid(beta) between phases.
struct K2Ub {
    static constexpr uint32_t kV      = 0;                      // [16,128] bf16
    static constexpr uint32_t kU      = kV + CHUNK * D * 2;      // [16,128] bf16
    static constexpr uint32_t kF32A   = kU + CHUNK * D * 2;      // [16,128] f32
    static constexpr uint32_t kF32B   = kF32A + CHUNK * D * 4;   // [16,128] f32
    static constexpr uint32_t kGTotal = kF32B + CHUNK * D * 4;   // [128] f32
    static constexpr uint32_t kBeta   = kGTotal + D * 4;         // [16] f32
    static constexpr uint32_t kStateA = kBeta + 2048;              // [128,128] f32 row block
    static constexpr uint32_t kStateB = kStateA + D * D * 4;
    // [D,D] bf16 staging for StateUbToGmT, which runs only when writing the
    // final state -- after the chunk loop, when kV/kU/kF32A/kF32B are dead.
    //
    // Overlaying it on those keeps kernel2 inside the 192 KB UB at larger
    // CHUNK, where a separate D*D*2 allocation does not fit. The front region
    // is only big enough from CHUNK=32 up, so at 16 it keeps its own space.
    static constexpr uint32_t kFront  = CHUNK * D * 2 * 2 + CHUNK * D * 4 * 2;
    static constexpr bool kNarrowFits = kFront >= D * D * 2;
    static constexpr uint32_t kNarrow = kNarrowFits ? kV : (kStateB + D * D * 4);
    static constexpr uint32_t kScalar =
        kNarrowFits ? (kStateB + D * D * 4) : (kStateB + D * D * 4 + D * D * 2);
    static constexpr uint32_t kEnd    = kScalar + 256;
};
static_assert(K2Ub::kEnd < ArchTag::UB_SIZE, "kernel2 UB budget exceeded");

// L1 map, AIC only. Two [16,128] operands plus the [128,128] state.
// Tight slots, deliberately. Widening these to 64 KB each made the cube fault,
// so the fractal loads depend on this packing more than on having slack --
// the opposite of what L0A/L0B/L0C wanted in kernel1.
struct K2L1 {
    static constexpr uint32_t kA     = 0;
    static constexpr uint32_t kB     = kA + CHUNK * D * 2;
    static constexpr uint32_t kState = kB + CHUNK * D * 2;
    static constexpr uint32_t kSmall = kState + D * D * 2;
    static constexpr uint32_t kEnd   = kSmall + 512;
};
static_assert(K2L1::kEnd < ArchTag::L1_SIZE, "kernel2 L1 budget exceeded");


// Explicit per-core-type buffers, constructed inside the phase that needs them.
//
// Catlass::Arch::Resource is deliberately not used: as a class member it is
// built on every launch regardless of whether the phase does any work, and
// kernel2 issues five launches per chunk. It also allocates UB unconditionally,
// which an AIC core has no use for.
struct K2AivBufs {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub;

    CATLASS_DEVICE
    K2AivBufs()
    {
        pipe.InitBuffer(ub, ArchTag::UB_SIZE);
    }

    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> Ub(uint32_t byteOffset)
    {
        return ub.Get<uint8_t>()[byteOffset].template ReinterpretCast<T>();
    }
};

struct K2AicBufs {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1;
    AscendC::TBuf<AscendC::TPosition::A2> l0a;
    AscendC::TBuf<AscendC::TPosition::B2> l0b;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c;

    CATLASS_DEVICE
    K2AicBufs()
    {
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
    CATLASS_DEVICE AscendC::LocalTensor<T> L0A() { return l0a.Get<T>(); }
    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L0B() { return l0b.Get<T>(); }
    template <class T>
    CATLASS_DEVICE AscendC::LocalTensor<T> L0C() { return l0c.Get<T>(); }
};

class FwdRecurrenceKernel {
public:
    using Params = FwdParams;

    CATLASS_DEVICE
    FwdRecurrenceKernel() {}

    // Phase entry points. Each is launched separately and ordered by the
    // stream; there are no cross-core flags, because those deadlock when the
    // kernel comes from a Python extension (docs/debugging-notes.md). The chunk
    // index arrives as an argument since the block index encodes only
    // (sequence, head).
    //
    // Every phase must state its core type: the compiler emits both a _mix_aic
    // and a _mix_aiv half for each kernel, and an unguarded body runs on both.

    struct SeqSpan {
        bool valid;
        int64_t bos;
        int seqLen;
        int tileBase;
        int nTiles;
        int headIdx;
        int seqIdx;
    };

    CATLASS_DEVICE
    SeqSpan Locate(Params const& params, bool isAiv)
    {
        SeqSpan sp{};
        const uint32_t coreIdx = isAiv
            ? AscendC::GetBlockIdx() / AscendC::GetSubBlockNum()
            : AscendC::GetBlockIdx();
        if (static_cast<int>(coreIdx) >= params.N * params.H) {
            return sp;
        }
        sp.headIdx = static_cast<int>(coreIdx) % params.H;
        sp.seqIdx = static_cast<int>(coreIdx) / params.H;
        ResolveSeq(params, sp.seqIdx, sp.bos, sp.seqLen, sp.tileBase);
        sp.nTiles = (sp.seqLen + CHUNK - 1) / CHUNK;
        sp.valid = true;
        return sp;
    }

    CATLASS_DEVICE
    void RunInitState(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (sp.valid) {
            K2AivBufs bufs;
            InitState(bufs, params, sp.seqIdx, sp.headIdx);
        }
    }

    CATLASS_DEVICE
    void RunBuildU(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        int len = sp.seqLen - chunk * CHUNK;
        if (len > CHUNK) {
            len = CHUNK;
        }
        K2AivBufs bufs;
        BuildU(bufs, params, sp.headIdx, sp.tileBase + chunk, sp.bos + chunk * CHUNK, len);
    }

    CATLASS_DEVICE
    void RunPreGemms(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        SeqSpan sp = Locate(params, false);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        K2AicBufs bufs;
        PreGemms(bufs, params, sp.seqIdx, sp.headIdx, sp.tileBase + chunk);
    }

    CATLASS_DEVICE
    void RunFinishOut(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        int len = sp.seqLen - chunk * CHUNK;
        if (len > CHUNK) {
            len = CHUNK;
        }
        K2AivBufs bufs;
        FinishOut(bufs, params, sp.headIdx, sp.tileBase + chunk, sp.bos + chunk * CHUNK, len);
    }

    CATLASS_DEVICE
    void RunPostGemms(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        SeqSpan sp = Locate(params, false);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        K2AicBufs bufs;
        PostGemms(bufs, params, sp.seqIdx, sp.headIdx, sp.tileBase + chunk);
    }

    CATLASS_DEVICE
    void RunFinishChunk(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        int len = sp.seqLen - chunk * CHUNK;
        if (len > CHUNK) {
            len = CHUNK;
        }
        const int tileIdx = sp.tileBase + chunk;
        K2AivBufs bufs;
        StoreOut(bufs, params, sp.headIdx, tileIdx, sp.bos + chunk * CHUNK, len);
        DecayState(bufs, params, sp.seqIdx, sp.headIdx, tileIdx);
    }

    // FinishChunk for chunk t, then BuildU for chunk t+1, in one launch.
    //
    // Kernel2 is 79% launch overhead (322 launches at ~5.7 us; all five phases
    // together cost 0.49 ms). The per-chunk chain
    //
    //     PreGemms (AIC) -> FinishOut (AIV) -> PostGemms (AIC) -> FinishChunk (AIV)
    //
    // alternates core type at every arrow, so those four boundaries are forced
    // while cross-core sync is unavailable. But BuildU of the next chunk is AIV
    // and already runs immediately after FinishChunk, so that one boundary buys
    // nothing.
    //
    // This does not reorder anything: BuildU still runs after DecayState has
    // produced the state it reads via StateToBf16, exactly as before. Only the
    // launch between them is gone. An earlier attempt hoisted BuildU out of the
    // loop entirely, which broke every multi-chunk shape precisely because it
    // moved that read away from the state that feeds it.
    CATLASS_DEVICE
    void RunFinishChunkAndBuildNext(Params const& params)
    {
        const int chunk = params.chunk_idx;
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (!sp.valid || chunk >= sp.nTiles) {
            return;
        }
        int len = sp.seqLen - chunk * CHUNK;
        if (len > CHUNK) {
            len = CHUNK;
        }
        const int tileIdx = sp.tileBase + chunk;

        K2AivBufs bufs;
        StoreOut(bufs, params, sp.headIdx, tileIdx, sp.bos + chunk * CHUNK, len);
        DecayState(bufs, params, sp.seqIdx, sp.headIdx, tileIdx);

        if (chunk + 1 < sp.nTiles) {
            // These were separate launches, so neither carries a barrier at its
            // boundary; back to back in one launch the UB buffers they share
            // need one.
            AscendC::PipeBarrier<PIPE_ALL>();

            int nlen = sp.seqLen - (chunk + 1) * CHUNK;
            if (nlen > CHUNK) {
                nlen = CHUNK;
            }
            BuildU(bufs, params, sp.headIdx, tileIdx + 1,
                   sp.bos + (chunk + 1) * CHUNK, nlen);
        }
    }

    // The whole recurrence in one kernel: chunk loop inside, AIC/AIV handoffs
    // by cross-core flag instead of launch boundary.
    //
    // Launch overhead is 54% of the pipeline and device-side (~5.2 us each,
    // 2051 launches at T=8192 H=64), and every one of kernel2's launch
    // boundaries existed only to hand off between core types. Cross-core sync
    // needs KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1) on the entry --
    // without a declared task type the runtime never provisions FFTS sync and
    // the first wait hangs, which is what made two earlier attempts conclude it
    // does not work.
    //
    // Both core types run this loop and alternate. The ordering is exactly what
    // the separate launches provided, so the arithmetic is unchanged.
    CATLASS_DEVICE
    void RunFusedAll(Params const& params)
    {
        AscendC::SetSyncBaseAddr(params.sync_base_addr);

        Catlass::Arch::CrossCoreFlag aivReady{1};
        Catlass::Arch::CrossCoreFlag aicReady{2};

        SeqSpan sp = Locate(params, g_coreType == AscendC::AIV);

        // Every block must keep handshaking until the longest sequence is done,
        // even after its own is finished: its peer is waiting on it. Blocks
        // past their own end set their flags and skip the work.
        const int maxChunks = params.total_tiles;

        if constexpr (g_coreType == AscendC::AIV) {
            // Both AIV subblocks run. Mode 0x2 pairs the two vector cores with
            // the one cube core inside an AI Core, so both must reach every
            // flag; the work split goes around the work, never around a flag.
            //
            // DecayState is 5.0 ms of kernel2's 6.65 ms of vector time, so it
            // and the bf16 cast are split by row range. The [CHUNK, D] phases
            // stay on subblock 0 for now -- they are 2.6 ms between them and
            // splitting them is a separate change.
            const uint32_t nsub = AscendC::GetSubBlockNum();
            const uint32_t sub = AscendC::GetSubBlockIdx();
            const int rowsPer = D / static_cast<int>(nsub);
            const int r0 = static_cast<int>(sub) * rowsPer;
            const bool lead = (sub == 0);

            K2AivBufs bufs;

            // The state lives here for the whole loop. InitState builds it in
            // kStateA and nothing after that reads or writes it through GM --
            // this block owns this (sequence, head) for every chunk, so the UB
            // copy is the only copy.
            auto stateResident = bufs.template Ub<float>(K2Ub::kStateA);

            if (sp.valid) {
                // Both subblocks build the whole state; each then maintains
                // only its own rows.
                InitState(bufs, params, sp.seqIdx, sp.headIdx);
                int len0 = sp.seqLen;
                if (len0 > CHUNK) {
                    len0 = CHUNK;
                }
                if (sp.nTiles > 0) {
                    BuildU(bufs, params, sp.headIdx, sp.tileBase, sp.bos, len0,
                           /*castState=*/false);
                    // Seed slot 3 from the resident state rather than letting
                    // BuildU read it back from GM into kStateA.
                    StateToBf16Resident(bufs, params, sp.tileBase, sp.headIdx,
                                        stateResident);
                }
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aivReady);

            for (int c = 0; c < maxChunks; ++c) {
                const bool live = sp.valid && c < sp.nTiles;
                int len = 0;
                int tileIdx = 0;
                if (live) {
                    len = sp.seqLen - c * CHUNK;
                    if (len > CHUNK) {
                        len = CHUNK;
                    }
                    tileIdx = sp.tileBase + c;
                }

                // The cube produced slot 4 / slot 6 for this chunk in its
                // previous turn (peeled for c=0, then as PreGemms(c+1) at the
                // end of the c-1 turn).
                Catlass::Arch::CrossCoreWaitFlag(aicReady);
                if (live && lead) {
                    FinishOut(bufs, params, sp.headIdx, tileIdx,
                              sp.bos + c * CHUNK, len);
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aivReady);

                // AIC is running PostGemms(c).
                Catlass::Arch::CrossCoreWaitFlag(aicReady);
                if (live) {
                    // Keep the balanced half-and-half split of DecayState -- it
                    // is 5.0 ms against StoreOut's 0.5, so handing a whole phase
                    // to each subblock is worse balance, measured at 18.01 ms
                    // against 16.57.
                    //
                    // Instead give the lead a smaller share of the rows and let
                    // StoreOut ride in the gap. StoreOut reads slots 6 and 8 and
                    // writes gmOut; DecayState reads slot 7 and the resident
                    // state; they do not touch each other.
                    //
                    // 8/20 measured, not guessed. The curve is a clean minimum:
                    //   5/20 16.86   6/20 16.65   7/20 16.45   8/20 16.23
                    //   9/20 16.34   10/20 16.59  11/20 16.81
                    // 10/20 is the even split, so the tilt is worth 0.36 ms and
                    // it is not simply "less for the lead is better".
                    const int leadRows = (D * 8) / 20;
                    const int myR0 = lead ? 0 : leadRows;
                    const int myRows = lead ? leadRows : (D - leadRows);
                    if (lead) {
                        StoreOut(bufs, params, sp.headIdx, tileIdx,
                                 sp.bos + c * CHUNK, len);
                    }
                    DecayStateResidentRows(bufs, params, sp.headIdx, tileIdx,
                                           stateResident, myR0, myRows);
                    if (c + 1 < sp.nTiles) {
                        // Same pairing the merged launch used: BuildU for the
                        // next chunk runs here, after DecayState has produced
                        // the state its StateToBf16 reads.
                        AscendC::PipeBarrier<PIPE_ALL>();
                        int nlen = sp.seqLen - (c + 1) * CHUNK;
                        if (nlen > CHUNK) {
                            nlen = CHUNK;
                        }
                        if (lead) {
                            BuildU(bufs, params, sp.headIdx, tileIdx + 1,
                                   sp.bos + (c + 1) * CHUNK, nlen,
                                   /*castState=*/false);
                        }
                        StateToBf16ResidentRows(bufs, params, tileIdx + 1,
                                                sp.headIdx, stateResident,
                                                myR0, myRows);
                    }
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(aivReady);
            }

            if (sp.valid && params.has_state_out != 0) {
                // The final state transposes across the whole tile, so each
                // subblock publishes its rows to GM first and the lead reads a
                // complete copy back. Once per sequence, not per chunk.
                {
                    // Same uneven split the loop used, so every row is covered.
                    const int leadRows = (D * 8) / 20;
                    PublishStateRows(params, sp.seqIdx, sp.headIdx, stateResident,
                                     lead ? 0 : leadRows,
                                     lead ? leadRows : (D - leadRows));
                }
                // StoreFinalState reads the state back from GM, and the last
                // chunk's DecayState wrote it from this same core moments ago.
                // As a separate launch that ordering was free; inside one
                // kernel the store must retire before the load is issued.
                AscendC::PipeBarrier<PIPE_ALL>();
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                if (lead) {
                    StoreFinalState(bufs, params, sp.seqIdx, sp.headIdx);
                }
            }
        } else {
            K2AicBufs bufs;

            // AIV is doing InitState and BuildU(0).
            Catlass::Arch::CrossCoreWaitFlag(aivReady);

            // One AIC turn per chunk instead of two.
            //
            // The counters account for ~4.8 ms of AIC pipe time and ~3.3 ms of
            // AIV time in a ~10.3 ms kernel; the missing ~2.2 ms is handshake
            // latency, 4 round trips per chunk over 256 chunks. Halving the
            // round trips attacks exactly that.
            //
            // It works because PostGemms(c) and PreGemms(c+1) can run back to
            // back: PostGemms(c) needs u(c) from FinishOut(c), and
            // PreGemms(c+1) needs slot 3 and slot 2 of tile c+1, both written
            // by the AIV's end-of-chunk phase. Slots are keyed by tileIdx, so
            // c and c+1 do not alias.
            //
            // PreGemms for chunk 0 is peeled out before the loop.
            if (sp.valid && sp.nTiles > 0) {
                PreGemms(bufs, params, sp.seqIdx, sp.headIdx, sp.tileBase);
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(aicReady);

            for (int c = 0; c < maxChunks; ++c) {
                const bool live = sp.valid && c < sp.nTiles;
                const int tileIdx = live ? (sp.tileBase + c) : 0;

                // AIV ran FinishOut(c); now PostGemms(c), then straight into
                // PreGemms(c+1) without handing back in between.
                Catlass::Arch::CrossCoreWaitFlag(aivReady);
                if (live) {
                    PostGemms(bufs, params, sp.seqIdx, sp.headIdx, tileIdx);
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(aicReady);

                // AIV runs the end-of-chunk phase, which writes slot 3 and
                // slot 2 for c+1; then the cube can run ahead.
                Catlass::Arch::CrossCoreWaitFlag(aivReady);
                if (live && c + 1 < sp.nTiles) {
                    PreGemms(bufs, params, sp.seqIdx, sp.headIdx, tileIdx + 1);
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(aicReady);
            }
        }
    }

    CATLASS_DEVICE
    void RunStoreFinalState(Params const& params)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        if (params.has_state_out == 0) {
            return;
        }
        SeqSpan sp = Locate(params, true);
        if (sp.valid) {
            K2AivBufs bufs;
            StoreFinalState(bufs, params, sp.seqIdx, sp.headIdx);
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
               WorkspaceSizes::SlotOffset(i);
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
    // Each phase is its own launch, so the stream orders them -- but a GM write
    // from one core type is not automatically visible to the next phase running
    // on the other.
    template <class T>
    CATLASS_DEVICE void FlushGm(AscendC::GlobalTensor<T> gm, int64_t elemOff, int count)
    {
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(gm[elemOff]);
        (void)count;
    }

    // GM initial_state / final_state are [value, key]; internally the state is
    // [key, value] because that is the orientation every GEMM here wants. These
    // two helpers do the transpose, and only run once per (sequence, head).
    //
    // DataCopyPad gathers a strided column: D blocks of one element with a
    // (D-1)-element gap, which is destination row r drawn from gm[:, r].

    // GM initial_state / final_state are [value, key]; internally the state is
    // [key, value], the orientation every GEMM here wants. These run once per
    // (sequence, head), not per chunk, so a scalar transpose is affordable.
    //
    // Deliberately not DataCopyPad: gathering a column with single-element
    // blocks looks natural but DataCopyPad pads each block to a 32-byte
    // datablock, so the values do not land contiguously. That is the bug that
    // corrupted the beta load, and here it would be repeated D times per
    // tensor. Copy whole, then transpose in fp32 with scalar reads.

    template <class T>
    CATLASS_DEVICE void StateGmToUbT(K2AivBufs& bufs, AscendC::LocalTensor<float> dst,
                                     AscendC::GlobalTensor<T> gm, int64_t base)
    {
        auto staging = bufs.template Ub<T>(K2Ub::kStateB);
        AscendC::DataCopy(staging, gm[base], D * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)6);

        // Widen into dst first -- the scalar unit cannot convert bf16 -- then
        // transpose dst in place. Swapping across the diagonal needs no second
        // [D,D] buffer, which matters: three of them would exceed the 192 KB UB.
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy(dst, staging.template ReinterpretCast<float>(), D * D);
        } else {
            AscendC::Cast(dst, staging, AscendC::RoundMode::CAST_NONE, D * D);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)6);
        for (int r = 0; r < D; ++r) {
            for (int c = r + 1; c < D; ++c) {
                const float a = dst.GetValue(r * D + c);
                const float b = dst.GetValue(c * D + r);
                dst.SetValue(r * D + c, b);
                dst.SetValue(c * D + r, a);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)6);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)6);
    }

    // Takes the fp32 state and narrows on output itself. It must not be handed
    // an already-narrowed bf16 copy: the caller's obvious place to put one is
    // kStateB, which is the very buffer this widens into, and Cast would then
    // read and overwrite the same address at two different widths.
    template <class T>
    CATLASS_DEVICE void StateUbToGmT(K2AivBufs& bufs, AscendC::GlobalTensor<T> gm,
                                     int64_t base, AscendC::LocalTensor<float> src)
    {
        auto wide = bufs.template Ub<float>(K2Ub::kStateB);
        AscendC::DataCopy(wide, src, D * D);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)7);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)7);
        for (int r = 0; r < D; ++r) {
            for (int c = r + 1; c < D; ++c) {
                const float a = wide.GetValue(r * D + c);
                const float b = wide.GetValue(c * D + r);
                wide.SetValue(r * D + c, b);
                wide.SetValue(c * D + r, a);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)7);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)7);

        // fp32 goes straight out of the transposed buffer; only bf16 needs a
        // narrow staging copy, and at half the width it fits the UB budget --
        // three [D,D] fp32 buffers would not.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)7);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy(gm[base], wide.template ReinterpretCast<T>(), D * D);
        } else {
            auto outb = bufs.template Ub<T>(K2Ub::kNarrow);
            AscendC::Cast(outb, wide, AscendC::RoundMode::CAST_RINT, D * D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)8);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)8);
            AscendC::DataCopy(gm[base], outb, D * D);
        }
    }

    CATLASS_DEVICE
    void InitState(K2AivBufs& bufs, Params const& params, int seqIdx, int headIdx)
    {
        auto sa = bufs.template Ub<float>(K2Ub::kStateA);
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
                StateGmToUbT<float>(bufs, sa, gmIn, src);
            } else {
                AscendC::GlobalTensor<BF16> gmIn;
                gmIn.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.initial_state));
                const int64_t src = (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D;
                // GM is [value, key]; sa is [key, value].
                StateGmToUbT<BF16>(bufs, sa, gmIn, src);
            }
        } else {
            AscendC::Duplicate(sa, 0.0f, D * D);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)0);
        AscendC::DataCopy(gmS[dst / 4], sa, D * D);
        FlushGm<float>(gmS, dst / 4, D * D);
    }

    // u = Mqk @ v - (INV @ k_dec) @ state, staged so the AIC can do both GEMMs
    // in one round. Here we only prepare v (and zero its tail rows) and the
    // bf16 copy of the state the cube will read.
    CATLASS_DEVICE
    // castState=false suppresses the trailing StateToBf16. The fused path needs
    // that: StateToBf16 reads the state from GM into kStateA, which is exactly
    // where the resident copy lives, so letting it run would overwrite the live
    // state with a stale GM one. It writes slot 3 itself instead.
    void BuildU(K2AivBufs& bufs, Params const& params, int headIdx, int tileIdx,
                int64_t tokenBase, int len, bool castState = true)
    {
        auto vb = bufs.template Ub<BF16>(K2Ub::kV);
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
        FlushGm<BF16>(wsB, Slot(params, tileIdx, headIdx, 2) / 2, CHUNK * D);

        // sigmoid(beta) for this chunk's rows, into slot 1 as fp32. FinishOut
        // needs it and runs as a separate launch, so it has to go through GM.
        // beta is [T_total, H]: this chunk's values start at the first token
        // and repeat every H elements.
        {
            auto braw = bufs.template Ub<BF16>(K2Ub::kBeta);
            auto bsig = bufs.template Ub<float>(K2Ub::kBeta + 64);
            auto bwrk = bufs.template Ub<float>(K2Ub::kBeta + 64 + 128);

            AscendC::GlobalTensor<BF16> gmBeta;
            gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.beta));
            const int64_t betaOff = tokenBase * params.H + headIdx;

            AscendC::Duplicate(braw, static_cast<BF16>(0), CHUNK);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)7);

            // beta is [T_total, H]: this chunk's rows are len*H contiguous
            // elements starting at the first token. Copy the block whole --
            // a strided DataCopyPad of one element per token does not work,
            // because DataCopyPad pads each block to a 32-byte datablock and
            // the values would land 16 elements apart with garbage between.
            const int nRaw = CHUNK * params.H;
            AscendC::DataCopy(braw, gmBeta[tokenBase * params.H], nRaw);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)7);
            auto bwide = bufs.template Ub<float>(K2Ub::kBeta + 64 + 256);
            AscendC::Cast(bwide, braw, AscendC::RoundMode::CAST_NONE, nRaw);
            AscendC::PipeBarrier<PIPE_V>();

            // Pick this head's column. Scalar float reads are fine; only
            // scalar bf16 casts are rejected, which is why the cast came first.
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)7);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)7);
            for (int r = 0; r < CHUNK; ++r) {
                bsig.SetValue(r, (r < len) ? bwide.GetValue(r * params.H + headIdx) : 0.0f);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)7);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)7);

            // sigmoid(x) = 1 / (1 + exp(-x)); no scalar exp on the aicore.
            AscendC::Muls(bwrk, bsig, -1.0f, CHUNK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(bwrk, bwrk, CHUNK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(bwrk, bwrk, 1.0f, CHUNK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(bsig, 1.0f, CHUNK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(bsig, bsig, bwrk, CHUNK);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::GlobalTensor<float> wsF;
            wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)7);
            AscendC::DataCopy(wsF[Slot(params, tileIdx, headIdx, 1) / 4], bsig, CHUNK);
        FlushGm<float>(wsF, Slot(params, tileIdx, headIdx, 1) / 4, CHUNK);
        }

        if (castState) {
            StateToBf16(bufs, params, tileIdx, headIdx);
        }
    }

    // The cube reads the state as bf16; the live copy is fp32 in GM.
    CATLASS_DEVICE
    void StateToBf16(K2AivBufs& bufs, Params const& params, int tileIdx, int headIdx)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const int seqIdx = static_cast<int>(coreIdx) / params.H;

        auto sf = bufs.template Ub<float>(K2Ub::kStateA);
        auto sb = bufs.template Ub<BF16>(K2Ub::kStateB);
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
        FlushGm<BF16>(wsB, Slot(params, tileIdx, headIdx, 3) / 2, D * D);
    }

    // After round 1 the cube left  slot4 = Mqk @ v  and  slot5 = (INV@k_dec)@state.
    // u = slot4 - slot5, then out = slot6 (q_dec@state) + Mqk @ u needs another
    // GEMM, so u goes back out for round 2.
    CATLASS_DEVICE
    void FinishOut(K2AivBufs& bufs, Params const& params, int headIdx, int tileIdx, int64_t tokenBase, int len)
    {
        auto a = bufs.template Ub<float>(K2Ub::kF32A);
        auto b = bufs.template Ub<float>(K2Ub::kF32B);
        auto ub = bufs.template Ub<BF16>(K2Ub::kU);

        AscendC::GlobalTensor<float> wsF;
        AscendC::GlobalTensor<BF16> wsB;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        // a = k_dec @ state (fp32 from the cube); b reused for v as fp32.
        AscendC::DataCopy(a, wsF[Slot(params, tileIdx, headIdx, 4) / 4], CHUNK * D);
        {
            auto vb = bufs.template Ub<BF16>(K2Ub::kV);
            AscendC::GlobalTensor<BF16> wsBv;
            wsBv.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));
            AscendC::DataCopy(vb, wsBv[Slot(params, tileIdx, headIdx, 2) / 2], CHUNK * D);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)8);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)8);
            AscendC::Cast(b, vb, AscendC::RoundMode::CAST_NONE, CHUNK * D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)2);

        // v_sub = (v - k_dec@state) * sigmoid(beta), row-broadcast.
        AscendC::Sub(a, b, a, CHUNK * D);
        AscendC::PipeBarrier<PIPE_V>();
        {
            auto bsig = bufs.template Ub<float>(K2Ub::kBeta + 64);
            AscendC::GlobalTensor<float> wsFb;
            wsFb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
            AscendC::DataCopy(bsig, wsFb[Slot(params, tileIdx, headIdx, 1) / 4], CHUNK);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)8);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)8);
            for (int r = 0; r < CHUNK; ++r) {
                const float bs = bsig.GetValue(r);
                AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)8);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)8);
                AscendC::Muls(a[r * D], a[r * D], bs, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::Cast(ub, a, AscendC::RoundMode::CAST_RINT, CHUNK * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)2);

        // u for round 2's two GEMMs (Mqk @ u and k_res^T @ u).
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 2) / 2], ub, CHUNK * D);
        FlushGm<BF16>(wsB, Slot(params, tileIdx, headIdx, 2) / 2, CHUNK * D);
        (void)tokenBase;
        (void)len;
    }

    // state = state * g_total + (k_res^T @ u), the latter left in slot 7 by the
    // cube in round 2 of the *previous* handshake half.
    // DecayState with the state already in UB and left there.
    //
    // state = state * g_total (row broadcast) + slot7, exactly as the GM
    // version, but without the 64 KB read at the top and the 64 KB write at
    // the bottom. Valid only because the fused kernel gives one AIV block
    // ownership of one (sequence, head) for every chunk, so the UB copy is the
    // only copy and nothing else can observe a stale GM one.
    //
    // slot 7 is still read from GM: the cube wrote it, and A2 has no L1<->UB
    // path, so that hop is forced.
    CATLASS_DEVICE
    void DecayStateResident(K2AivBufs& bufs, Params const& params, int headIdx,
                            int tileIdx, AscendC::LocalTensor<float> sf)
    {
        auto upd = bufs.template Ub<float>(K2Ub::kStateB);
        auto gt = bufs.template Ub<float>(K2Ub::kGTotal);

        AscendC::GlobalTensor<float> wsF;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        AscendC::DataCopy(upd, wsF[Slot(params, tileIdx, headIdx, 7) / 4], D * D);
        AscendC::DataCopy(gt, wsF[Ws(params, tileIdx, headIdx,
                                     WorkspaceOffsets::kGTotal) / 4], D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)3);

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
    }

    // DecayState over a row range only, so the two AIV subblocks can split it.
    //
    // Each subblock keeps the full [D, D] state in its own UB -- they both run
    // InitState -- but maintains only rows [r0, r0+rows). Indexing by the
    // global row therefore works unchanged, and nothing reads the half a
    // subblock does not own.
    CATLASS_DEVICE
    void DecayStateResidentRows(K2AivBufs& bufs, Params const& params, int headIdx,
                                int tileIdx, AscendC::LocalTensor<float> sf,
                                int r0, int rows)
    {
        auto upd = bufs.template Ub<float>(K2Ub::kStateB);
        auto gt = bufs.template Ub<float>(K2Ub::kGTotal);

        AscendC::GlobalTensor<float> wsF;
        wsF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));

        AscendC::DataCopy(upd[r0 * D],
                          wsF[Slot(params, tileIdx, headIdx, 7) / 4 + r0 * D],
                          rows * D);
        AscendC::DataCopy(gt, wsF[Ws(params, tileIdx, headIdx,
                                     WorkspaceOffsets::kGTotal) / 4], D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)3);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)3);

        for (int r = r0; r < r0 + rows; ++r) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)0);
            const float g = gt.GetValue(r);
            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)0);
            AscendC::Muls(sf[r * D], sf[r * D], g, D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(sf[r0 * D], sf[r0 * D], upd[r0 * D], rows * D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // The bf16 cast of a row range, likewise split across subblocks.
    CATLASS_DEVICE
    void StateToBf16ResidentRows(K2AivBufs& bufs, Params const& params, int tileIdx,
                                 int headIdx, AscendC::LocalTensor<float> sf,
                                 int r0, int rows)
    {
        auto sb = bufs.template Ub<BF16>(K2Ub::kStateB);
        AscendC::GlobalTensor<BF16> wsB;
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::Cast(sb[r0 * D], sf[r0 * D], AscendC::RoundMode::CAST_RINT, rows * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 3) / 2 + r0 * D],
                          sb[r0 * D], rows * D);
        FlushGm<BF16>(wsB, Slot(params, tileIdx, headIdx, 3) / 2 + r0 * D, rows * D);
    }

    // Publish a row range of the resident state back to the state slot in GM,
    // so the final-state path (which transposes across the whole tile) can read
    // a complete copy.
    CATLASS_DEVICE
    void PublishStateRows(Params const& params, int seqIdx, int headIdx,
                          AscendC::LocalTensor<float> sf, int r0, int rows)
    {
        AscendC::GlobalTensor<float> gmS;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
        const int64_t off = StateOff(params, seqIdx, headIdx) / 4 + r0 * D;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)4);
        AscendC::DataCopy(gmS[off], sf[r0 * D], rows * D);
        FlushGm<float>(gmS, off, rows * D);
    }

    // Cast a UB-resident state to bf16 for the cube, without reading GM.
    CATLASS_DEVICE
    void StateToBf16Resident(K2AivBufs& bufs, Params const& params, int tileIdx,
                             int headIdx, AscendC::LocalTensor<float> sf)
    {
        // kStateB, not kNarrow: kNarrow now overlays the front buffers, which
        // are live mid-loop. kStateB holds DecayState's `upd`, dead by here,
        // and this is the same buffer StateToBf16 uses for the same purpose.
        // sf is kStateA, so there is no aliasing with the source.
        auto sb = bufs.template Ub<BF16>(K2Ub::kStateB);
        AscendC::GlobalTensor<BF16> wsB;
        wsB.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        AscendC::Cast(sb, sf, AscendC::RoundMode::CAST_RINT, D * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)1);
        AscendC::DataCopy(wsB[Slot(params, tileIdx, headIdx, 3) / 2], sb, D * D);
        FlushGm<BF16>(wsB, Slot(params, tileIdx, headIdx, 3) / 2, D * D);
    }

    // Write a UB-resident state out as the final state, without reading GM.
    CATLASS_DEVICE
    void StoreFinalStateResident(K2AivBufs& bufs, Params const& params, int seqIdx,
                                 int headIdx, AscendC::LocalTensor<float> sf)
    {
        const int64_t dst = (static_cast<int64_t>(seqIdx) * params.H + headIdx) * D * D;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
        if (params.state_fp32 != 0) {
            AscendC::GlobalTensor<float> gmFs;
            gmFs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.final_state));
            StateUbToGmT<float>(bufs, gmFs, dst, sf);
        } else {
            AscendC::GlobalTensor<BF16> gmFs;
            gmFs.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.final_state));
            StateUbToGmT<BF16>(bufs, gmFs, dst, sf);
        }
    }

    CATLASS_DEVICE
    void DecayState(K2AivBufs& bufs, Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        auto sf = bufs.template Ub<float>(K2Ub::kStateA);
        auto upd = bufs.template Ub<float>(K2Ub::kStateB);
        auto gt = bufs.template Ub<float>(K2Ub::kGTotal);

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
        // g_total is read below with GetValue, i.e. by the scalar unit. MTE2_V
        // orders the load against the vector unit only; the scalar unit needs
        // its own MTE2_S or the first read can see stale UB.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((event_t)3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>((event_t)3);

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
        FlushGm<float>(gmS, stateOff / 4, D * D);
    }

    // out = slot6 + slot8, both fp32 from the cube; rounded once to bf16 as the
    // reference does, and only real rows are written.
    CATLASS_DEVICE
    void StoreOut(K2AivBufs& bufs, Params const& params, int headIdx, int tileIdx, int64_t tokenBase, int len)
    {
        auto a = bufs.template Ub<float>(K2Ub::kF32A);
        auto b = bufs.template Ub<float>(K2Ub::kF32B);
        auto ob = bufs.template Ub<BF16>(K2Ub::kU);

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
    void StoreFinalState(K2AivBufs& bufs, Params const& params, int seqIdx, int headIdx)
    {
        auto sf = bufs.template Ub<float>(K2Ub::kStateA);
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
            StateUbToGmT<float>(bufs, gmFs, dst, sf);
        } else {
            AscendC::GlobalTensor<BF16> gmFs;
            gmFs.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.final_state));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)5);
            // sb is [key, value]; GM wants [value, key].
            StateUbToGmT<BF16>(bufs, gmFs, dst, sf);
        }
    }

    // ---------------- AIC ----------------
    // Round 1: slot4 = Mqk @ v, slot5 = (INV @ k_dec) @ state, slot6 = q_dec @ state.
    CATLASS_DEVICE
    void PreGemms(K2AicBufs& bufs, Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        const int64_t mqk = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kMqk);
        const int64_t inv = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kINV);
        const int64_t kdec = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kKDecayed);
        const int64_t qdec = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kQDecayed);
        const int64_t v = Slot(params, tileIdx, headIdx, 2);
        const int64_t state = Slot(params, tileIdx, headIdx, 3);

        // s4 = k_dec @ state, the delta-rule prediction to subtract from v.
        Gemm(bufs, params, kdec, state, Slot(params, tileIdx, headIdx, 4), CHUNK, D, D, false);
        // s6 = q_dec @ state, the first half of out.
        // Same state as operand B, already in L1 from the call above.
        Gemm(bufs, params, qdec, state, Slot(params, tileIdx, headIdx, 6), CHUNK, D, D, false,
             /*loadB=*/false);
        (void)seqIdx;
        (void)mqk;
        (void)inv;
        (void)v;
    }

    // Round 2: slot8 = Mqk @ u, slot7 = k_res^T @ u.
    CATLASS_DEVICE
    void PostGemms(K2AicBufs& bufs, Params const& params, int seqIdx, int headIdx, int tileIdx)
    {
        const int64_t mqk = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kMqk);
        const int64_t kres = Ws(params, tileIdx, headIdx, WorkspaceOffsets::kKRestored);
        const int64_t u = Slot(params, tileIdx, headIdx, 2);

        // u = INV @ v_sub, written as bf16 so the next two GEMMs can read it.
        const int64_t uu = Slot(params, tileIdx, headIdx, 0);
        Gemm(bufs, params, Ws(params, tileIdx, headIdx, WorkspaceOffsets::kINV), u,
             uu, CHUNK, D, CHUNK, true);
        Gemm(bufs, params, mqk, uu, Slot(params, tileIdx, headIdx, 8), CHUNK, D, CHUNK, false);
        // k_res^T[128,16] @ u[16,128]: the transpose comes from reading k_res's
        // RowMajor bytes as ColumnMajor, not from any data movement.
        GemmAt(bufs, params, kres, uu, Slot(params, tileIdx, headIdx, 7));
        (void)seqIdx;
    }

    // C[m,n] = A[m,k] @ B[k,n], all operands bf16 in GM scratch.
    CATLASS_DEVICE
    // loadB=false reuses whatever is already in the L1 B slot.
    //
    // PreGemms runs k_dec @ state and q_dec @ state back to back with the same
    // state as operand B, and each call reloaded all 32 KB of it. The cube is
    // memory-bound here -- aic_mte2_ratio 0.244 against aic_mac_ratio 0.02, so
    // it is fetching operands rather than computing -- and the state is the
    // bulk of what it fetches.
    void Gemm(K2AicBufs& bufs, Params const& params, int64_t aByte, int64_t bByte, int64_t cByte,
              int m, int n, int k, bool bf16Out, bool loadB = true)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = bufs.template L1<BF16>(K2L1::kA);
        auto l1B = bufs.template L1<BF16>(K2L1::kState);
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)0);

        LoadNd2Nz(bufs, l1A, gm, aByte, m, k);   // A: RowMajor -> zN, feeds L0A
        if (loadB) {
            LoadBGmToL1zZ(l1B, gm, bByte, k, n); // B: RowMajor -> zZ, feeds L0B
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = false;
        ld.repeatTimes = (m / 16) * (k / 16);
        AscendC::LoadData(l0A, l1A, ld);
        LoadBL1ToL0B(l0B, l1B, k, n);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)0);

        AscendC::MmadParams mp;
        mp.m = m;
        mp.n = n;
        mp.k = k;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        // The MMAD reads L0A/L0B; the next GEMM's LoadData overwrites them.
        // Without M_MTE1 between the two, MTE1 can start writing L0B while the
        // cube is still reading it -- reported as "L0B read/write conflict in
        // the MTE (same address)". These helpers are called back to back with
        // the same L0 buffers, so the barrier belongs right after the Mmad.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)5);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)0);

        Drain(bufs, params, l0C, cByte, m, n, bf16Out);
    }

    // C[128,128] = A^T[128,16] @ B[16,128], A stored RowMajor [16,128].
    CATLASS_DEVICE
    void GemmAt(K2AicBufs& bufs, Params const& params, int64_t aByte, int64_t bByte, int64_t cByte)
    {
        AscendC::GlobalTensor<BF16> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ BF16*>(params.workspace));

        auto l1A = bufs.template L1<BF16>(K2L1::kA);
        auto l1B = bufs.template L1<BF16>(K2L1::kB);
        auto l0A = bufs.template L0A<BF16>();
        auto l0B = bufs.template L0B<BF16>();
        auto l0C = bufs.template L0C<float>();

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>((event_t)1);

        // k_res is [CHUNK, D] RowMajor; its transpose is the same bytes viewed
        // as ColumnMajor [D, CHUNK]. Nd2Nz describes a ColumnMajor source by
        // its columns: nValue is the column count, dValue the column length,
        // srcDValue the column pitch. The reversed form asks for D spans of
        // CHUNK at a D pitch and runs off the end of a CHUNK*D buffer -- the
        // same out-of-bounds read that faulted kernel1's LoadBt.
        //
        // For nZ [D, CHUNK]: stride(1) = colsRound * C0 = 256, stride(2) = 16,
        // so dstNzC0Stride = 256/16 = CHUNK and dstNzNStride = 1.
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
        LoadBGmToL1zZ(l1B, gm, bByte, CHUNK, D);   // B: RowMajor -> zZ

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)1);

        AscendC::LoadData2DParams la;
        // Matches CopyL1ToL0A<nZ -> zZ> exactly: one repeat per destination
        // column fractal, issued once per destination row fractal, rather than
        // a single call with repeatTimes covering both. The two coincide only
        // if the fractals happen to be contiguous in both, which is not
        // something to rely on.
        //   A is [D, CHUNK]: dst zZ stride(1) = 16*16 = 256, src nZ the same.
        la.startIndex = 0;
        la.srcStride = 1;                       // src nZ stride(3)/256 = 1
        la.dstGap = 0;                          // dst zZ stride(3)/256 - 1 = 0
        la.ifTranspose = true;
        la.repeatTimes = CHUNK / C0_NUM_PER_FRACTAL;   // ceil(16/16) = 1
        for (int i = 0; i < D / C0_NUM_PER_FRACTAL; ++i) {
            AscendC::LoadData(l0A[i * 256], l1A[i * 256], la);
        }

        LoadBL1ToL0B(l0B, l1B, CHUNK, D);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((event_t)1);

        AscendC::MmadParams mp;
        mp.m = D;
        mp.n = D;
        mp.k = CHUNK;
        mp.cmatrixInitVal = true;
        AscendC::Mmad(l0C, l0A, l0B, mp);

        // The MMAD reads L0A/L0B; the next GEMM's LoadData overwrites them.
        // Without M_MTE1 between the two, MTE1 can start writing L0B while the
        // cube is still reading it -- reported as "L0B read/write conflict in
        // the MTE (same address)". These helpers are called back to back with
        // the same L0 buffers, so the barrier belongs right after the Mmad.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((event_t)5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((event_t)5);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((event_t)1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((event_t)1);

        Drain(bufs, params, l0C, cByte, D, D, false);
    }

    // ---- B operand path: GM RowMajor -> L1 zZ -> L0B nZ (transposing) ----
    //
    // L0B accepts only zZ (transposing) or nZ (not). A RowMajor B has to go
    // through zZ; loading it as zN and issuing ifTranspose = false is not a
    // supported combination and silently produces the wrong operand.

    CATLASS_DEVICE
    void LoadBGmToL1zZ(AscendC::LocalTensor<BF16> dst, AscendC::GlobalTensor<BF16> src,
                       int64_t byteOff, int k, int n)
    {
        const int colsRound = (n + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL
                              * C0_NUM_PER_FRACTAL;
        AscendC::Nd2NzParams p;
        p.ndNum = static_cast<uint16_t>(k / C0_NUM_PER_FRACTAL);
        p.nValue = static_cast<uint16_t>(C0_NUM_PER_FRACTAL);
        p.dValue = static_cast<uint16_t>(n);
        p.srcNdMatrixStride = static_cast<uint16_t>(C0_NUM_PER_FRACTAL * n);
        p.srcDValue = static_cast<uint16_t>(n);
        p.dstNzC0Stride = static_cast<uint16_t>(C0_NUM_PER_FRACTAL);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = static_cast<uint16_t>(colsRound * C0_NUM_PER_FRACTAL);
        AscendC::DataCopy(dst, src[byteOff / 2], p);
    }

    CATLASS_DEVICE
    void LoadBL1ToL0B(AscendC::LocalTensor<BF16> l0B, AscendC::LocalTensor<BF16> l1B,
                      int k, int n)
    {
        const int colsRound = (n + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL
                              * C0_NUM_PER_FRACTAL;
        const int fractalRows = (k + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL;
        const int stride1 = colsRound * C0_NUM_PER_FRACTAL;

        AscendC::LoadData2DParams ld;
        ld.startIndex = 0;
        ld.repeatTimes = static_cast<uint16_t>((n + C0_NUM_PER_FRACTAL - 1)
                                               / C0_NUM_PER_FRACTAL);
        ld.srcStride = 1;
        ld.dstGap = 0;
        ld.ifTranspose = true;
        for (int i = 0; i < fractalRows; ++i) {
            AscendC::LoadData(l0B[i * stride1], l1B[i * stride1], ld);
        }
    }

    CATLASS_DEVICE
    void LoadNd2Nz(K2AicBufs& bufs, AscendC::LocalTensor<BF16> dst, AscendC::GlobalTensor<BF16> src,
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
    void Drain(K2AicBufs& bufs, Params const& params, AscendC::LocalTensor<float> l0C, int64_t cByte,
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
        // Gemm and GemmAt already open with a self-consuming FIX_MTE2 barrier
        // before reloading L1, so setting the events again here only leaks
        // them: two per GEMM, never waited on. That is the same imbalance that
        // stalled kernel1's Neumann chain.
    }
};

}  // namespace flash_kda
