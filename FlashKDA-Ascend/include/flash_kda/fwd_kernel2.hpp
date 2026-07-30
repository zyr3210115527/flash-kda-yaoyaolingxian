/**
 * FlashKDA Ascend — Kernel 2: Recurrence (Full Implementation)
 *
 * AIC-dominant kernel with pingpong pipeline:
 *   1. Load initial state
 *   2. Per-chunk loop:
 *      a. AIC: Dual GEMM (k_decayed@state, q_decayed@state)
 *      b. AIC: u = INV @ ((v - k_decayed@state) * beta)
 *      c. AIC: out = q_decayed@state + Mqk @ u
 *      d. AIV: state = state * exp(g_total) + k_restored^T @ u
 *      e. Store out to GM
 *   3. Store final state
 *
 * Grid: (N * H) — one AIC+AIV core per (sequence, head)
 * Pipeline: 2-stage pingpong with HardEvent synchronization
 * AIC-AIV sync: CrossCoreSetFlag/WaitFlag
 *
 * Memory layout:
 *   L1 (512KB): pingpong buffers for workspace intermediates + state
 *   L0A (64KB): A operand for MMAD
 *   L0B (64KB): B operand for MMAD
 *   L0C (128KB): accumulator
 *   UB (192KB): element-wise computation buffers
 */

#pragma once

#include "flash_kda/layout.hpp"
#include "flash_kda/utils.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace Catlass::Arch;

namespace flash_kda {

// ============================================================
// Constants
// ============================================================
constexpr int INPUT_STAGES = 2;
constexpr int K2_MMAD_READY_ID = 1;    // AIC -> AIV
constexpr int K2_ELEM_READY_ID  = 2;    // AIV -> AIC

// MMAD tile sizes
constexpr int MMAD_M = 16;
constexpr int MMAD_N = 16;
constexpr int MMAD_K = 16;

// Number of k-blocks for D=128: 128/16 = 8
constexpr int K_BLOCKS = D / MMAD_K;
// Number of n-blocks for D=128: 128/16 = 8
constexpr int N_BLOCKS = D / MMAD_N;

// ============================================================
// L1 buffer layout for Kernel 2 (pingpong, 2 stages)
// ============================================================
// Per stage workspace:
//   k_decayed [CHUNK, D] bf16 zN = 4096 bytes
//   q_decayed [CHUNK, D] bf16 zN = 4096 bytes
//   k_restored [CHUNK, D] bf16 zN = 4096 bytes
//   v [CHUNK, D] bf16 zN = 4096 bytes
//   INV [CHUNK, CHUNK] bf16 zN = 512 bytes
//   Mqk [CHUNK, CHUNK] bf16 zN = 512 bytes
//   g_total [D] fp32 = 512 bytes
//   beta [32] bf16 = 64 bytes
// Per stage total: ~17KB
// 2 stages: ~34KB
// Plus state [D, D] bf16 zN = 32768 bytes (shared across stages)
// Total L1: ~67KB

constexpr int L1_STATE_OFF = 0;
constexpr int L1_STATE_SIZE = D * D * sizeof(half);  // 32768

// Stage 0 offsets
constexpr int L1_S0_OFF = L1_STATE_OFF + L1_STATE_SIZE;
constexpr int L1_S0_KD_OFF   = L1_S0_OFF;               // [CHUNK, D] bf16 zN = 4096
constexpr int L1_S0_QD_OFF   = L1_S0_KD_OFF + 4096;     // [CHUNK, D] bf16 zN = 4096
constexpr int L1_S0_KR_OFF   = L1_S0_QD_OFF + 4096;     // [CHUNK, D] bf16 zN = 4096
constexpr int L1_S0_V_OFF    = L1_S0_KR_OFF + 4096;      // [CHUNK, D] bf16 zN = 4096
constexpr int L1_S0_INV_OFF  = L1_S0_V_OFF + 4096;       // [CHUNK, CHUNK] bf16 zN = 512
constexpr int L1_S0_MQK_OFF  = L1_S0_INV_OFF + 512;      // [CHUNK, CHUNK] bf16 zN = 512
constexpr int L1_S0_GT_OFF   = L1_S0_MQK_OFF + 512;      // [D] fp32 = 512
constexpr int L1_S0_BETA_OFF = L1_S0_GT_OFF + 512;       // [32] bf16 = 64
constexpr int L1_S0_SIZE = 4096 * 4 + 512 * 3 + 64;      // ~17KB

// Stage 1 offsets
constexpr int L1_S1_OFF = L1_S0_OFF + L1_S0_SIZE;
// Same layout as Stage 0 (offsets relative to L1_S1_OFF)

// ============================================================
// UB buffer layout for AIV element-wise operations
// ============================================================
constexpr int UB2_OUT_OFF     = 0;                          // [CHUNK, D] bf16 = 4096
constexpr int UB2_U_ACC_OFF   = UB2_OUT_OFF + 4096;         // [CHUNK, D] bf16 = 4096 (k_decayed@state)
constexpr int UB2_OUT_ACC_OFF = UB2_U_ACC_OFF + 4096;       // [CHUNK, D] bf16 = 4096 (q_decayed@state)
constexpr int UB2_U_OFF       = UB2_OUT_ACC_OFF + 4096;     // [CHUNK, D] bf16 = 4096 (u result)
constexpr int UB2_GT_OFF      = UB2_U_OFF + 4096;           // [D] fp32 = 512
constexpr int UB2_BETA_OFF    = UB2_GT_OFF + 512;           // [32] bf16 = 64
constexpr int UB2_STATE_OFF   = UB2_BETA_OFF + 64;          // [D, D] bf16 = 32768 (for elem-wise)
// Total UB: ~49KB (well within 192KB)

// ============================================================
// Kernel 2 class
// ============================================================
class FwdRecurrenceKernel {
public:
    __aicore__ FwdRecurrenceKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ void operator()(const FwdParams& params);

private:
    CrossCoreFlag mmadReady{K2_MMAD_READY_ID};
    CrossCoreFlag elemReady{K2_ELEM_READY_ID};

    // ============================================================
    // AIC: Load workspace tile from GM to L1
    // ============================================================
    __aicore__ void load_workspace_to_l1(
        const FwdParams& params, int ws_idx, int64_t bos, int t,
        int head_idx, int stage
    ) {
        int64_t ws_base = (int64_t)ws_idx * WorkspaceSizes::kPerTile;
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        // Nd2NzParams for GM RowMajor [CHUNK, D] → L1 zN format
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = CHUNK;
        nd2nz.dValue = D;
        nd2nz.srcDValue = D;          // RowMajor row stride
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzNStride = D / 16;  // zN N-stride in C0 blocks
        nd2nz.dstNzC0Stride = 1;
        nd2nz.dstNzMatrixStride = 0;

        // Load k_decayed [CHUNK, D] bf16: GM RowMajor → L1 zN
        {
            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_KD_OFF - L1_S0_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKDecayed),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nz);
        }

        // Load q_decayed [CHUNK, D] bf16
        {
            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_QD_OFF - L1_S0_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kQDecayed),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nz);
        }

        // Load k_restored [CHUNK, D] bf16
        {
            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_KR_OFF - L1_S0_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKRestored),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nz);
        }

        // Load v [CHUNK, D] bf16 from GM (v is an input, not in workspace)
        // v is stored as [B, T, H, D] in GM, need [CHUNK, D] for this tile
        {
            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_V_OFF - L1_S0_OFF);
            int64_t t_offset = bos + t * CHUNK;
            GlobalTensor<half> gm_v;
            gm_v.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.v) + head_idx * params.T_total * D + t_offset * D,
                CHUNK * D);
            DataCopy(l1_dst, gm_v, nd2nz);
        }

        // Load INV [CHUNK, CHUNK] bf16: GM RowMajor → L1 zN
        {
            Nd2NzParams nd2nz_small;
            nd2nz_small.ndNum = 1;
            nd2nz_small.nValue = CHUNK;
            nd2nz_small.dValue = CHUNK;
            nd2nz_small.srcDValue = CHUNK;
            nd2nz_small.srcNdMatrixStride = 0;
            nd2nz_small.dstNzNStride = 1;  // 16/16 = 1
            nd2nz_small.dstNzC0Stride = 1;
            nd2nz_small.dstNzMatrixStride = 0;

            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_INV_OFF - L1_S0_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kINV),
                CHUNK * CHUNK);
            DataCopy(l1_dst, gm_src, nd2nz_small);
        }

        // Load Mqk [CHUNK, CHUNK] bf16
        {
            Nd2NzParams nd2nz_small;
            nd2nz_small.ndNum = 1;
            nd2nz_small.nValue = CHUNK;
            nd2nz_small.dValue = CHUNK;
            nd2nz_small.srcDValue = CHUNK;
            nd2nz_small.srcNdMatrixStride = 0;
            nd2nz_small.dstNzNStride = 1;
            nd2nz_small.dstNzC0Stride = 1;
            nd2nz_small.dstNzMatrixStride = 0;

            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_MQK_OFF - L1_S0_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kMqk),
                CHUNK * CHUNK);
            DataCopy(l1_dst, gm_src, nd2nz_small);
        }

        // Load g_total [D] fp32: GM → L1 (1D copy, no fractal)
        {
            LocalTensor<float> l1_dst = l1Buf_.GetBufferByByte<float>(s_off + L1_S0_GT_OFF - L1_S0_OFF);
            GlobalTensor<float> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.workspace + ws_base + WorkspaceOffsets::kGTotal),
                D);
            DataCopy(l1_dst, gm_src, D);
        }

        // Load beta [CHUNK] bf16 from GM (beta is an input, not in workspace)
        {
            LocalTensor<half> l1_dst = l1Buf_.GetBufferByByte<half>(s_off + L1_S0_BETA_OFF - L1_S0_OFF);
            int64_t t_offset = bos + t * CHUNK;
            GlobalTensor<half> gm_beta;
            gm_beta.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.beta) + head_idx * params.T_total + t_offset,
                CHUNK);
            DataCopy(l1_dst, gm_beta, CHUNK);
        }

        SetFlag<HardEvent::MTE2_MTE1>(stage);
    }

    // ============================================================
    // AIC: Load initial state from GM to L1
    // ============================================================
    __aicore__ void load_initial_state_aic(
        const FwdParams& params, int seq_idx, int head_idx
    ) {
        if (!params.has_state_in) {
            // Zero-initialize state in L1
            LocalTensor<half> state_l1 = l1Buf_.GetBufferByByte<half>(L1_STATE_OFF);
            Duplicate(state_l1, half(0), D * D);
            return;
        }

        // Load state [D, D] bf16 from GM to L1 zN format
        int64_t state_off = (seq_idx * params.H + head_idx) * D * D;
        LocalTensor<half> state_l1 = l1Buf_.GetBufferByByte<half>(L1_STATE_OFF);

        if (params.state_fp32) {
            // FP32 state: load fp32 to UB, Cast to bf16, copy to L1 zN
            // Step 1: Load fp32 state [D, D] from GM to UB
            LocalTensor<float> fp32_ub = ubBuf_.GetBufferByByte<float>(UB2_STATE_OFF);
            GlobalTensor<float> gm_fp32_state;
            gm_fp32_state.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.initial_state) + state_off,
                D * D);
            DataCopy(fp32_ub, gm_fp32_state, D * D);
            PipeBarrier<PIPE_V>();

            // Step 2: Cast fp32 → bf16 in UB
            LocalTensor<half> bf16_ub = ubBuf_.GetBufferByByte<half>(UB2_STATE_OFF);
            Cast(bf16_ub, fp32_ub, RoundMode::CAST_RINT, D * D);
            PipeBarrier<PIPE_V>();

            // Step 3: Copy bf16 from UB to L1 (simple copy for state)
            DataCopy(state_l1, bf16_ub, D * D);
        } else {
            // BF16 state: load directly to L1 zN
            GlobalTensor<half> gm_state;
            gm_state.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.initial_state) + state_off,
                D * D);
            Nd2NzParams nzParams;
            nzParams.ndNum = 1;
            nzParams.nValue = D;
            nzParams.dValue = D;
            nzParams.srcDValue = D;
            nzParams.srcNdMatrixStride = 0;
            nzParams.dstNzNStride = D / 16;
            nzParams.dstNzC0Stride = 1;
            nzParams.dstNzMatrixStride = 0;
            DataCopy(state_l1, gm_state, nzParams);
        }
    }

    // ============================================================
    // AIC: Dual GEMM — k_decayed@state and q_decayed@state
    // ============================================================
    // k_decayed [CHUNK, D] @ state [D, D] → u_acc [CHUNK, D]
    //   = sum over k=0..7 of k_decayed[16, 16k:16(k+1)] @ state[16k:16(k+1), 16n:16(n+1)]
    //   for each n-block n=0..7
    //
    // q_decayed [CHUNK, D] @ state [D, D] → out_acc [CHUNK, D]
    //   Same structure
    //
    // Results stored to UB for AIV element-wise ops
    __aicore__ void dual_gemm_aic(int stage) {
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        // LoadData params for A operand (k_decayed/q_decayed, no transpose)
        LoadData2DParams ldParamsA;
        ldParamsA.startIndex = 0;
        ldParamsA.repeatTimes = 1;   // 1 fractal repeat (16 elements)
        ldParamsA.srcStride = 1;
        ldParamsA.sid = 0;
        ldParamsA.dstGap = 0;
        ldParamsA.ifTranspose = false;
        ldParamsA.addrMode = 0;

        // LoadData params for B operand (state, transposed for B side)
        LoadData2DParams ldParamsB;
        ldParamsB.startIndex = 0;
        ldParamsB.repeatTimes = 1;
        ldParamsB.srcStride = 1;
        ldParamsB.sid = 0;
        ldParamsB.dstGap = 0;
        ldParamsB.ifTranspose = true;   // state is B operand, needs transpose
        ldParamsB.addrMode = 0;

        // ---- k_decayed @ state → u_acc ----
        for (int n = 0; n < N_BLOCKS; ++n) {
            LocalTensor<float> l0C = l0CBuf_.GetBufferByByte<float>(0);

            for (int k = 0; k < K_BLOCKS; ++k) {
                // Load k_decayed[16, 16k:16(k+1)] from L1 → L0A
                // zN format: k-th D-block offset = l1_zn_block_off(0, k, D)
                LocalTensor<half> l0A = l0ABuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1A = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_KD_OFF - L1_S0_OFF + l1_zn_block_off(0, k, D));
                WaitFlag<HardEvent::M_MTE1>(stage);
                LoadData(l0A, l1A, ldParamsA);

                // Load state[16k:16(k+1), 16n:16(n+1)] from L1 → L0B
                // zN format: block [k, n] offset = l1_zn_block_off(k, n, D)
                LocalTensor<half> l0B = l0BBuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1B = l1Buf_.GetBufferByByte<half>(
                    L1_STATE_OFF + l1_zn_block_off(k, n, D));
                LoadData(l0B, l1B, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                // MMAD: l0C += l0A @ l0B
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = MMAD_M;
                mmadParams.n = MMAD_N;
                mmadParams.k = MMAD_K;
                mmadParams.cmatrixInitVal = (k == 0);
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(stage);
            }

            // Fixpipe: L0C → UB (u_acc for this n-block)
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            LocalTensor<half> u_acc_ub = ubBuf_.GetBufferByByte<half>(UB2_U_ACC_OFF + n * 16 * sizeof(half));
            FixpipeParamsV220 fixParams;
            fixParams.nSize = MMAD_N;
            fixParams.mSize = MMAD_M;
            fixParams.srcStride = 1;
            fixParams.dstStride = D;  // RowMajor stride in UB
            fixParams.quantPre = QuantMode_t::F322BF16;  // fp32 L0C → bf16 UB
            fixParams.reluEn = false;
            fixParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(u_acc_ub, l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }

        // ---- q_decayed @ state → out_acc ----
        for (int n = 0; n < N_BLOCKS; ++n) {
            LocalTensor<float> l0C = l0CBuf_.GetBufferByByte<float>(0);

            for (int k = 0; k < K_BLOCKS; ++k) {
                LocalTensor<half> l0A = l0ABuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1A = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_QD_OFF - L1_S0_OFF + l1_zn_block_off(0, k, D));
                WaitFlag<HardEvent::M_MTE1>(stage);
                LoadData(l0A, l1A, ldParamsA);

                LocalTensor<half> l0B = l0BBuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1B = l1Buf_.GetBufferByByte<half>(
                    L1_STATE_OFF + l1_zn_block_off(k, n, D));
                LoadData(l0B, l1B, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = MMAD_M;
                mmadParams.n = MMAD_N;
                mmadParams.k = MMAD_K;
                mmadParams.cmatrixInitVal = (k == 0);
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(stage);
            }

            // Fixpipe: L0C → UB (out_acc for this n-block)
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            LocalTensor<half> out_acc_ub = ubBuf_.GetBufferByByte<half>(UB2_OUT_ACC_OFF + n * 16 * sizeof(half));
            FixpipeParamsV220 fixParams;
            fixParams.nSize = MMAD_N;
            fixParams.mSize = MMAD_M;
            fixParams.srcStride = 1;
            fixParams.dstStride = D;
            fixParams.quantPre = QuantMode_t::F322BF16;
            fixParams.reluEn = false;
            fixParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(out_acc_ub, l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    // ============================================================
    // AIC: u = INV @ ((v - u_acc) * beta)
    // ============================================================
    // After dual GEMM, u_acc = k_decayed @ state is in UB2_U_ACC_OFF
    // v is in L1 (loaded from GM)
    // beta is in L1
    //
    // Step 1 (AIV): v_sub = (v - u_acc) * sigmoid(beta) → store to UB
    // Step 2 (AIC): u = INV @ v_sub → MMAD per n-block
    //
    // This requires AIC-AIV sync between steps.
    // Alternative: AIC does INV @ v first, then subtract INV @ u_acc
    //   u = INV @ v_sub = INV @ v - INV @ u_acc
    //   But u_acc = k_decayed @ state, so INV @ u_acc is part of the recurrence
    //   This doesn't simplify things.
    //
    // Best approach: AIV computes v_sub, then AIC does INV @ v_sub
    __aicore__ void compute_u_aic(int stage) {
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        // INV [CHUNK, CHUNK] @ v_sub [CHUNK, D] → u [CHUNK, D]
        // Per n-block: u[:, 16n:16(n+1)] = INV @ v_sub[:, 16n:16(n+1)]
        // This is a matrix-vector product where INV is 16x16 and v_sub is 16x16 per block

        LoadData2DParams ldParamsA;
        ldParamsA.startIndex = 0;
        ldParamsA.repeatTimes = 1;
        ldParamsA.srcStride = 1;
        ldParamsA.sid = 0;
        ldParamsA.dstGap = 0;
        ldParamsA.ifTranspose = false;
        ldParamsA.addrMode = 0;

        LoadData2DParams ldParamsB;
        ldParamsB.startIndex = 0;
        ldParamsB.repeatTimes = 1;
        ldParamsB.srcStride = 1;
        ldParamsB.sid = 0;
        ldParamsB.dstGap = 0;
        ldParamsB.ifTranspose = true;
        ldParamsB.addrMode = 0;

        for (int n = 0; n < N_BLOCKS; ++n) {
            LocalTensor<float> l0C = l0CBuf_.GetBufferByByte<float>(0);

            // INV is 16x16, v_sub[:, 16n:16(n+1)] is 16x16
            // Single MMAD: u[:, 16n:16(n+1)] = INV @ v_sub[:, 16n:16(n+1)]
            {
                LocalTensor<half> l0A = l0ABuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1A = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_INV_OFF - L1_S0_OFF);
                WaitFlag<HardEvent::M_MTE1>(stage);
                LoadData(l0A, l1A, ldParamsA);

                // v_sub is in UB (AIV computed it)
                // On Ascend, L0B can only be loaded from L1, not UB
                // So: copy v_sub from UB → L1 temp, then L1 → L0B
                LocalTensor<half> v_sub_ub = ubBuf_.GetBufferByByte<half>(UB2_U_OFF + n * 16 * sizeof(half));
                LocalTensor<half> v_sub_l1 = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_V_OFF - L1_S0_OFF);  // reuse V L1 space as temp
                DataCopy(v_sub_l1, v_sub_ub, CHUNK * 16);

                LocalTensor<half> l0B = l0BBuf_.GetBufferByByte<half>(0);
                LoadData(l0B, v_sub_l1, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = MMAD_M;
                mmadParams.n = MMAD_N;
                mmadParams.k = MMAD_K;
                mmadParams.cmatrixInitVal = true;
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(stage);
            }

            // Fixpipe: L0C → UB (u result for this n-block)
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            LocalTensor<half> u_ub = ubBuf_.GetBufferByByte<half>(UB2_U_OFF + n * 16 * sizeof(half));
            FixpipeParamsV220 fixParams;
            fixParams.nSize = MMAD_N;
            fixParams.mSize = MMAD_M;
            fixParams.srcStride = 1;
            fixParams.dstStride = D;
            fixParams.quantPre = QuantMode_t::F322BF16;
            fixParams.reluEn = false;
            fixParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(u_ub, l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    // ============================================================
    // AIC: out = q_decayed@state + Mqk @ u
    // ============================================================
    __aicore__ void compute_out_aic(int stage) {
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        // out_acc = q_decayed @ state is already in UB2_OUT_ACC_OFF
        // Mqk [CHUNK, CHUNK] @ u [CHUNK, D] → add to out_acc
        // Per n-block: out[:, 16n:16(n+1)] += Mqk @ u[:, 16n:16(n+1)]

        LoadData2DParams ldParamsA;
        ldParamsA.startIndex = 0;
        ldParamsA.repeatTimes = 1;
        ldParamsA.srcStride = 1;
        ldParamsA.sid = 0;
        ldParamsA.dstGap = 0;
        ldParamsA.ifTranspose = false;
        ldParamsA.addrMode = 0;

        LoadData2DParams ldParamsB;
        ldParamsB.startIndex = 0;
        ldParamsB.repeatTimes = 1;
        ldParamsB.srcStride = 1;
        ldParamsB.sid = 0;
        ldParamsB.dstGap = 0;
        ldParamsB.ifTranspose = true;
        ldParamsB.addrMode = 0;

        for (int n = 0; n < N_BLOCKS; ++n) {
            LocalTensor<float> l0C = l0CBuf_.GetBufferByByte<float>(0);

            // Mqk @ u[:, 16n:16(n+1)]
            {
                LocalTensor<half> l0A = l0ABuf_.GetBufferByByte<half>(0);
                LocalTensor<half> l1A = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_MQK_OFF - L1_S0_OFF);
                WaitFlag<HardEvent::M_MTE1>(stage);
                LoadData(l0A, l1A, ldParamsA);

                // u is in UB, copy to L1 temp then load
                LocalTensor<half> u_ub = ubBuf_.GetBufferByByte<half>(UB2_U_OFF + n * 16 * sizeof(half));
                LocalTensor<half> u_l1 = l1Buf_.GetBufferByByte<half>(
                    s_off + L1_S0_V_OFF - L1_S0_OFF);  // reuse V L1 space
                DataCopy(u_l1, u_ub, CHUNK * 16);

                LocalTensor<half> l0B = l0BBuf_.GetBufferByByte<half>(0);
                LoadData(l0B, u_l1, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = MMAD_M;
                mmadParams.n = MMAD_N;
                mmadParams.k = MMAD_K;
                mmadParams.cmatrixInitVal = true;
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(stage);
            }

            // Fixpipe: L0C → UB (Mqk@u result)
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            LocalTensor<half> mqk_u_ub = ubBuf_.GetBufferByByte<half>(UB2_OUT_OFF + n * 16 * sizeof(half));
            FixpipeParamsV220 fixParams;
            fixParams.nSize = MMAD_N;
            fixParams.mSize = MMAD_M;
            fixParams.srcStride = 1;
            fixParams.dstStride = D;
            fixParams.quantPre = QuantMode_t::F322BF16;
            fixParams.reluEn = false;
            fixParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(mqk_u_ub, l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }

        // AIV adds out_acc + Mqk@u and stores to GM
        // Copy out_acc and Mqk@u to UB for AIV
        // out_acc is already in UB2_OUT_ACC_OFF
        // Mqk@u is in UB2_OUT_OFF
        // AIV will: out = out_acc + Mqk@u
    }

    // ============================================================
    // AIC: State update GEMM — k_restored^T @ u
    // ============================================================
    // state = state * exp(g_total) + k_restored^T @ u
    // k_restored^T [D, CHUNK] @ u [CHUNK, D] → [D, D]
    __aicore__ void state_update_gemm_aic(int stage) {
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        LoadData2DParams ldParamsA;
        ldParamsA.startIndex = 0;
        ldParamsA.repeatTimes = 1;
        ldParamsA.srcStride = 1;
        ldParamsA.sid = 0;
        ldParamsA.dstGap = 0;
        ldParamsA.ifTranspose = true;   // k_restored^T: transpose A
        ldParamsA.addrMode = 0;

        LoadData2DParams ldParamsB;
        ldParamsB.startIndex = 0;
        ldParamsB.repeatTimes = 1;
        ldParamsB.srcStride = 1;
        ldParamsB.sid = 0;
        ldParamsB.dstGap = 0;
        ldParamsB.ifTranspose = true;   // u: transpose B
        ldParamsB.addrMode = 0;

        // For each output block [16m:16(m+1), 16n:16(n+1)] of state:
        for (int m = 0; m < D / MMAD_M; ++m) {
            for (int n = 0; n < N_BLOCKS; ++n) {
                LocalTensor<float> l0C = l0CBuf_.GetBufferByByte<float>(0);

                // k_restored^T [D, CHUNK] @ u [CHUNK, D]
                // = sum over k=0..CHUNK/16-1 of
                //   k_restored[16k:16(k+1), 16m:16(m+1)]^T @ u[16k:16(k+1), 16n:16(n+1)]
                for (int k = 0; k < CHUNK / MMAD_K; ++k) {
                    // Load k_restored[16k:16(k+1), 16m:16(m+1)] to L0A (transposed)
                    // k_restored is [CHUNK, D] in zN: block [k, m] offset
                    LocalTensor<half> l0A = l0ABuf_.GetBufferByByte<half>(0);
                    LocalTensor<half> l1A = l1Buf_.GetBufferByByte<half>(
                        s_off + L1_S0_KR_OFF - L1_S0_OFF + l1_zn_block_off(k, m, D));
                    WaitFlag<HardEvent::M_MTE1>(stage);
                    LoadData(l0A, l1A, ldParamsA);

                    // Load u[16k:16(k+1), 16n:16(n+1)] to L0B
                    // u is in UB (RowMajor), copy to L1 first
                    LocalTensor<half> u_ub = ubBuf_.GetBufferByByte<half>(
                        UB2_U_OFF + (k * D + n * 16) * sizeof(half));
                    LocalTensor<half> u_l1 = l1Buf_.GetBufferByByte<half>(
                        s_off + L1_S0_V_OFF - L1_S0_OFF);  // reuse V L1 space
                    DataCopy(u_l1, u_ub, 16 * 16);

                    LocalTensor<half> l0B = l0BBuf_.GetBufferByByte<half>(0);
                    LoadData(l0B, u_l1, ldParamsB);
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                    MmadParams mmadParams;
                    mmadParams.m = MMAD_M;
                    mmadParams.n = MMAD_N;
                    mmadParams.k = MMAD_K;
                    mmadParams.cmatrixInitVal = (k == 0);
                    mmadParams.unitFlag = 0;
                    Mmad(l0C, l0A, l0B, mmadParams);
                    SetFlag<HardEvent::M_MTE1>(stage);
                }

                // Fixpipe: L0C → UB (state update for this block)
                // Then AIV adds to state * exp(g_total)
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                LocalTensor<half> state_update_ub = ubBuf_.GetBufferByByte<half>(
                    UB2_STATE_OFF + (m * D + n * 16) * sizeof(half));
                FixpipeParamsV220 fixParams;
                fixParams.nSize = MMAD_N;
                fixParams.mSize = MMAD_M;
                fixParams.srcStride = 1;
                fixParams.dstStride = D;
                fixParams.quantPre = QuantMode_t::F322BF16;
                fixParams.reluEn = false;
                fixParams.unitFlag = 0;
                Fixpipe<half, float, CFG_ROW_MAJOR>(state_update_ub, l0C, fixParams);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }
    }

    // ============================================================
    // AIV: Element-wise operations per chunk
    // ============================================================
    // 1. v_sub = (v - u_acc) * sigmoid(beta)  → stored to UB2_U_OFF
    // 2. out = out_acc + Mqk@u  → stored to UB2_OUT_OFF
    // 3. state = state * exp(g_total) + k_restored^T @ u  → update L1 state
    __aicore__ void elemwise_aiv(
        const FwdParams& params, int stage, int head_idx,
        int64_t bos, int t, int actual_len
    ) {
        int s_off = (stage == 0) ? L1_S0_OFF : L1_S1_OFF;

        // 1. v_sub = (v - u_acc) * sigmoid(beta)
        // u_acc is in UB2_U_ACC_OFF [CHUNK, D] bf16
        // v needs to be loaded from GM to UB
        // beta needs to be loaded from GM to UB
        LocalTensor<half> v_ub = ubBuf_.GetBufferByByte<half>(UB2_U_OFF);  // reuse u space for v_sub
        LocalTensor<half> u_acc_ub = ubBuf_.GetBufferByByte<half>(UB2_U_ACC_OFF);
        LocalTensor<half> beta_ub = ubBuf_.GetBufferByByte<half>(UB2_BETA_OFF);

        // Load v [CHUNK, D] from GM
        {
            int64_t t_offset = bos + t * CHUNK;
            GlobalTensor<half> gm_v;
            gm_v.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.v) + head_idx * params.T_total * D + t_offset * D,
                CHUNK * D);
            // Load v to a temp location first
            LocalTensor<half> v_tmp = ubBuf_.GetBufferByByte<half>(UB2_OUT_OFF);  // temp
            DataCopy(v_tmp, gm_v, CHUNK * D);
            PipeBarrier<PIPE_V>();

            // v_sub = v - u_acc (element-wise)
            Sub(v_ub, v_tmp, u_acc_ub, CHUNK * D);
        }

        // Load beta [CHUNK] from GM
        {
            int64_t t_offset = bos + t * CHUNK;
            GlobalTensor<half> gm_beta;
            gm_beta.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.beta) + head_idx * params.T_total + t_offset,
                CHUNK);
            DataCopy(beta_ub, gm_beta, CHUNK);
            PipeBarrier<PIPE_V>();
        }

        // Apply sigmoid(beta) and multiply: v_sub *= sigmoid(beta)
        // sigmoid via tanh: process each row
        for (int row = 0; row < CHUNK; ++row) {
            half beta_val = beta_ub.GetValue(row);
            float beta_f = static_cast<float>(beta_val);
            float sig_beta = sigmoid_tanh_approx(beta_f);
            Muls(v_ub + row * D, v_ub + row * D, half(sig_beta), D);
        }
        PipeBarrier<PIPE_V>();

        // 2. out = out_acc + Mqk@u
        // out_acc is in UB2_OUT_ACC_OFF, Mqk@u is in UB2_OUT_OFF
        // After AIC computes Mqk@u, it's in UB2_OUT_OFF
        // We need to wait for AIC to finish compute_out_aic first
        // This is handled by the CrossCore sync in the main loop

        LocalTensor<half> out_ub = ubBuf_.GetBufferByByte<half>(UB2_OUT_OFF);
        LocalTensor<half> out_acc_ub = ubBuf_.GetBufferByByte<half>(UB2_OUT_ACC_OFF);
        Add(out_ub, out_acc_ub, out_ub, CHUNK * D);
        PipeBarrier<PIPE_V>();

        // 3. state = state * exp(g_total) + k_restored^T @ u
        // g_total is in L1, need to load to UB
        LocalTensor<float> g_total_ub = ubBuf_.GetBufferByByte<float>(UB2_GT_OFF);
        {
            LocalTensor<float> g_total_l1 = l1Buf_.GetBufferByByte<float>(
                s_off + L1_S0_GT_OFF - L1_S0_OFF);
            DataCopy(g_total_ub, g_total_l1, D);
            PipeBarrier<PIPE_V>();
        }

        // exp(g_total)
        Exp(g_total_ub, g_total_ub, D);
        PipeBarrier<PIPE_V>();

        // state * exp(g_total): load state from L1, multiply, add k_restored^T@u
        // k_restored^T@u is in UB2_STATE_OFF (AIC computed it)
        // For each block of state:
        for (int m = 0; m < D / MMAD_M; ++m) {
            for (int n = 0; n < N_BLOCKS; ++n) {
                // Load state[16m:16(m+1), 16n:16(n+1)] from L1 zN to UB
                LocalTensor<half> state_block_ub = ubBuf_.GetBufferByByte<half>(
                    UB2_STATE_OFF + 32768 + (m * D + n * 16) * sizeof(half));  // temp after state_update
                LocalTensor<half> state_l1 = l1Buf_.GetBufferByByte<half>(
                    L1_STATE_OFF + l1_zn_block_off(m, n, D));
                DataCopy(state_block_ub, state_l1, 16 * 16);
                PipeBarrier<PIPE_V>();

                // Multiply each row by exp(g_total[16m+row]) (broadcast across columns)
                // state[i, j] *= exp(g_total[i]), where i = 16m + row
                for (int row = 0; row < 16; ++row) {
                    float gt_val = g_total_ub.GetValue(m * 16 + row);
                    Muls(state_block_ub + row * 16, state_block_ub + row * 16, half(gt_val), 16);
                }
                PipeBarrier<PIPE_V>();

                // Add k_restored^T@u
                LocalTensor<half> update_ub = ubBuf_.GetBufferByByte<half>(
                    UB2_STATE_OFF + (m * D + n * 16) * sizeof(half));
                Add(state_block_ub, state_block_ub, update_ub, 16 * 16);
                PipeBarrier<PIPE_V>();

                // Write back to L1 zN
                DataCopy(state_l1, state_block_ub, 16 * 16);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // ============================================================
    // AIV: Store output to GM
    // ============================================================
    __aicore__ void store_output_aiv(
        const FwdParams& params, int t, int64_t bos, int head_idx, int actual_len
    ) {
        LocalTensor<half> out_ub = ubBuf_.GetBufferByByte<half>(UB2_OUT_OFF);

        int64_t t_offset = bos + t * CHUNK;
        int64_t gm_off = head_idx * params.T_total * D + t_offset * D;

        if (actual_len < CHUNK) {
            // Tail tile: store only actual_len rows
            for (int row = 0; row < actual_len; ++row) {
                GlobalTensor<half> gm_out;
                gm_out.SetGlobalBuffer(params.out + gm_off + row * D, D);
                DataCopy(gm_out, out_ub[row * D], D);
            }
        } else {
            // Full tile: bulk copy [CHUNK, D]
            GlobalTensor<half> gm_out;
            gm_out.SetGlobalBuffer(params.out + gm_off, CHUNK * D);
            DataCopy(gm_out, out_ub, CHUNK * D);
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIC: Store final state to GM
    // ============================================================
    __aicore__ void store_final_state_aic(
        const FwdParams& params, int seq_idx, int head_idx
    ) {
        if (!params.has_state_out) return;

        int64_t state_off = (seq_idx * params.H + head_idx) * D * D;

        if (params.state_fp32) {
            // Convert bf16 state in L1 to fp32, then store to GM
            // Step 1: Copy bf16 state from L1 to UB
            LocalTensor<half> state_l1 = l1Buf_.GetBufferByByte<half>(L1_STATE_OFF);
            LocalTensor<half> bf16_ub = ubBuf_.GetBufferByByte<half>(UB2_STATE_OFF);
            DataCopy(bf16_ub, state_l1, D * D);
            PipeBarrier<PIPE_V>();

            // Step 2: Cast bf16 → fp32 in UB
            LocalTensor<float> fp32_ub = ubBuf_.GetBufferByByte<float>(UB2_STATE_OFF);
            Cast(fp32_ub, bf16_ub, RoundMode::CAST_NONE, D * D);
            PipeBarrier<PIPE_V>();

            // Step 3: Store fp32 from UB to GM RowMajor
            GlobalTensor<float> gm_fp32_state;
            gm_fp32_state.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.final_state) + state_off,
                D * D);
            DataCopy(gm_fp32_state, fp32_ub, D * D);
        } else {
            // Store bf16 state from L1 to GM RowMajor
            // Step 1: Copy bf16 state from L1 to UB
            LocalTensor<half> state_l1 = l1Buf_.GetBufferByByte<half>(L1_STATE_OFF);
            LocalTensor<half> bf16_ub = ubBuf_.GetBufferByByte<half>(UB2_STATE_OFF);
            DataCopy(bf16_ub, state_l1, D * D);
            PipeBarrier<PIPE_V>();

            // Step 2: Store bf16 from UB to GM RowMajor
            GlobalTensor<half> gm_state;
            gm_state.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.final_state) + state_off,
                D * D);
            DataCopy(gm_state, bf16_ub, D * D);
        }
    }

    // Buffer handles
    TBuf<TPosition::A1>       l1Buf_;
    TBuf<TPosition::A2>       l0ABuf_;
    TBuf<TPosition::B2>       l0BBuf_;
    TBuf<TPosition::CO1>      l0CBuf_;
    TBuf<TPosition::VECCALC>  ubBuf_;
};

// ============================================================
// AIC entry point — Cube (MMAD) pipeline
// ============================================================
template <>
__aicore__ void FwdRecurrenceKernel::operator()<AIC>(const FwdParams& params) {
    TPipe pipe;
    pipe.InitBuffer(l1Buf_, L1_SIZE);
    pipe.InitBuffer(l0ABuf_, L0A_SIZE);
    pipe.InitBuffer(l0BBuf_, L0B_SIZE);
    pipe.InitBuffer(l0CBuf_, L0C_SIZE);

    // Initialize HardEvent flags
    for (int i = 0; i < INPUT_STAGES; i++) {
        SetFlag<HardEvent::M_MTE1>(i);
        SetFlag<HardEvent::MTE1_MTE2>(i);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);

    // Decode grid coordinates
    int linear_idx = GetBlockIdx();
    int seq_idx = linear_idx / params.H;
    int head_idx = linear_idx % params.H;

    // Compute sequence boundaries
    int64_t bos = 0, eos = 0;
    int tile_base = 0;
    if (params.is_varlen) {
        // Read cu_seqlens[seq_idx] and cu_seqlens[seq_idx+1] from GM
        // Scalar GM read via DataCopy to UB temp
        LocalTensor<int64_t> cu_tmp = ubBuf_.GetBufferByByte<int64_t>(UB2_GT_OFF);
        {
            GlobalTensor<int64_t> g_cu;
            g_cu.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + seq_idx, 1);
            DataCopy(cu_tmp, g_cu, 1);
            PipeBarrier<PIPE_V>();
            bos = cu_tmp.GetValue(0);

            g_cu.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + seq_idx + 1, 1);
            DataCopy(cu_tmp, g_cu, 1);
            PipeBarrier<PIPE_V>();
            eos = cu_tmp.GetValue(0);
        }
        // tile_base: sum of tiles in sequences before seq_idx
        // This requires scanning cu_seqlens[0..seq_idx], which is expensive.
        // Alternative: precompute tile_base on host and pass via params.
        // For now, scan on device (only once at kernel entry).
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            GlobalTensor<int64_t> g_cu0, g_cu1;
            g_cu0.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i, 1);
            DataCopy(cu_tmp, g_cu0, 1);
            PipeBarrier<PIPE_V>();
            int64_t s0 = cu_tmp.GetValue(0);
            g_cu1.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i + 1, 1);
            DataCopy(cu_tmp, g_cu1, 1);
            PipeBarrier<PIPE_V>();
            int64_t s1 = cu_tmp.GetValue(0);
            int slen = int(s1 - s0);
            tile_base += (slen + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = params.T_total / params.N;
        bos = (int64_t)seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }

    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;

    // Phase 0: Load initial state
    load_initial_state_aic(params, seq_idx, head_idx);

    // Phase 1: Pingpong pipeline main loop
    for (int t = 0; t < t_tiles; ++t) {
        int stage = t % INPUT_STAGES;
        int ws_idx = head_idx * params.total_tiles + tile_base + t;

        // Preload workspace to L1 (GM → L1)
        WaitFlag<HardEvent::MTE1_MTE2>(stage);
        load_workspace_to_l1(params, ws_idx, bos, t, head_idx, stage);
        SetFlag<HardEvent::MTE2_MTE1>(stage);

        // Wait for L1 data
        WaitFlag<HardEvent::MTE2_MTE1>(stage);

        // Phase 1a: Dual GEMM (k_decayed@state → u_acc, q_decayed@state → out_acc)
        dual_gemm_aic(stage);

        // Signal AIV: u_acc and out_acc ready in UB
        // AIV computes: v_sub = (v - u_acc) * sigmoid(beta)
        CrossCoreSetFlag<0x2, PIPE_FIX>(mmadReady);

        // Wait for AIV: v_sub computed in UB
        CrossCoreWaitFlag(elemReady);

        // Phase 1b: Compute u = INV @ v_sub
        compute_u_aic(stage);

        // Phase 1c: Compute Mqk @ u → add to out_acc
        compute_out_aic(stage);

        // Phase 1d: State update GEMM (k_restored^T @ u)
        state_update_gemm_aic(stage);

        // Signal AIV: all MMAD results ready
        CrossCoreSetFlag<0x2, PIPE_FIX>(mmadReady);

        // Wait for AIV: element-wise state update + output store done
        CrossCoreWaitFlag(elemReady);

        // Advance pipeline
        SetFlag<HardEvent::M_MTE1>(stage);
    }

    // Phase 2: Store final state
    store_final_state_aic(params, seq_idx, head_idx);

    // Release HardEvent flags
    for (int i = 0; i < INPUT_STAGES; i++) {
        WaitFlag<HardEvent::M_MTE1>(i);
        WaitFlag<HardEvent::MTE1_MTE2>(i);
    }
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

// ============================================================
// AIV entry point — Vector (element-wise) operations
// ============================================================
template <>
__aicore__ void FwdRecurrenceKernel::operator()<AIV>(const FwdParams& params) {
    TPipe pipe;
    pipe.InitBuffer(ubBuf_, UB_SIZE);

    int linear_idx = GetBlockIdx();
    int seq_idx = linear_idx / params.H;
    int head_idx = linear_idx % params.H;

    int64_t bos = 0, eos = 0;
    int tile_base = 0;
    if (params.is_varlen) {
        // Read cu_seqlens[seq_idx] and cu_seqlens[seq_idx+1] from GM
        LocalTensor<int64_t> cu_tmp = ubBuf_.GetBufferByByte<int64_t>(UB2_GT_OFF);
        {
            GlobalTensor<int64_t> g_cu;
            g_cu.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + seq_idx, 1);
            DataCopy(cu_tmp, g_cu, 1);
            PipeBarrier<PIPE_V>();
            bos = cu_tmp.GetValue(0);

            g_cu.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + seq_idx + 1, 1);
            DataCopy(cu_tmp, g_cu, 1);
            PipeBarrier<PIPE_V>();
            eos = cu_tmp.GetValue(0);
        }
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            GlobalTensor<int64_t> g_cu0, g_cu1;
            g_cu0.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i, 1);
            DataCopy(cu_tmp, g_cu0, 1);
            PipeBarrier<PIPE_V>();
            int64_t s0 = cu_tmp.GetValue(0);
            g_cu1.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i + 1, 1);
            DataCopy(cu_tmp, g_cu1, 1);
            PipeBarrier<PIPE_V>();
            int64_t s1 = cu_tmp.GetValue(0);
            int slen = int(s1 - s0);
            tile_base += (slen + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = params.T_total / params.N;
        bos = (int64_t)seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }

    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;

    for (int t = 0; t < t_tiles; ++t) {
        int stage = t % INPUT_STAGES;
        int actual_len = min(CHUNK, seq_len - t * CHUNK);

        // Wait for AIC: u_acc and out_acc ready in UB
        CrossCoreWaitFlag(mmadReady);

        // Element-wise: v_sub, out, state update
        elemwise_aiv(params, stage, head_idx, bos, t, actual_len);

        // Signal AIC: v_sub ready
        CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady);

        // Wait for AIC: Mqk@u and state_update ready
        CrossCoreWaitFlag(mmadReady);

        // Final output assembly and store
        // out = out_acc + Mqk@u (already done in elemwise_aiv)
        // Store output to GM
        store_output_aiv(params, t, bos, head_idx, actual_len);

        // Signal AIC: output stored, state updated
        CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady);
    }
}

}  // namespace flash_kda
