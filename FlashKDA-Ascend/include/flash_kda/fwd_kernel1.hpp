/**
 * FlashKDA Ascend — Kernel 1: Prepare (Full Implementation)
 *
 * AIV-dominant kernel that computes per-tile intermediates for KDA.
 *
 * Data flow:
 *   AIV: GM → UB (load) → vector ops → UB → GM (store workspace)
 *   AIC: GM → L1 (Nd2Nz) → L0A/L0B (LoadData) → MMAD → L0C → UB (Fixpipe)
 *
 * AIC-AIV sync: CrossCoreSetFlag/WaitFlag (2 rounds)
 *   Round 1: AIV signals ELEM_READY → AIC computes L, Mqk → signals MMA_READY
 *   Round 2: AIV applies mask, constructs (I-L) → signals ELEM_READY
 *            → AIC computes Neumann INV → signals MMA_READY
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
constexpr int K1_ELEM_READY_ID = 1;
constexpr int K1_MMA_READY_ID  = 2;

// UB buffer byte sizes
constexpr int UB_Q_SIZE       = CHUNK * D * sizeof(half);      // 4096
constexpr int UB_K_SIZE       = CHUNK * D * sizeof(half);      // 4096
constexpr int UB_G_BF16_SIZE  = CHUNK * D * sizeof(half);      // 4096
constexpr int UB_G_FP32_SIZE  = CHUNK * D * sizeof(float);     // 8192
constexpr int UB_BETA_SIZE    = 32 * sizeof(half);              // 64
constexpr int UB_DT_BIAS_SIZE = D * sizeof(float);              // 512
constexpr int UB_G_TOTAL_SIZE = D * sizeof(float);              // 512
constexpr int UB_LM_SIZE      = CHUNK * CHUNK * sizeof(half);   // 512

// UB offsets — Phase A (inputs)
constexpr int UB_Q_OFF        = 0;
constexpr int UB_K_OFF        = UB_Q_OFF + UB_Q_SIZE;
constexpr int UB_G_BF16_OFF  = UB_K_OFF + UB_K_SIZE;
constexpr int UB_G_FP32_OFF  = UB_G_BF16_OFF + UB_G_BF16_SIZE;
constexpr int UB_BETA_OFF    = UB_G_FP32_OFF + UB_G_FP32_SIZE;
constexpr int UB_DT_BIAS_OFF = UB_BETA_OFF + UB_BETA_SIZE;

// UB offsets — Phase B (intermediates, reuse Phase A space via union)
constexpr int UB_K_DECAYED_OFF  = 0;
constexpr int UB_Q_DECAYED_OFF  = UB_K_DECAYED_OFF + UB_Q_SIZE;
constexpr int UB_K_INV_OFF      = UB_Q_DECAYED_OFF + UB_Q_SIZE;
constexpr int UB_K_RESTORED_OFF = UB_K_INV_OFF + UB_Q_SIZE;
constexpr int UB_G_TOTAL_OFF    = UB_K_RESTORED_OFF + UB_Q_SIZE;
constexpr int UB_L_OFF          = UB_G_TOTAL_OFF + UB_G_TOTAL_SIZE;
constexpr int UB_INV_OFF        = UB_L_OFF + UB_LM_SIZE;
constexpr int UB_MQK_OFF        = UB_INV_OFF + UB_LM_SIZE;

// Temp fp32 buffer for L2 norm (reuses g_fp32 space)
constexpr int UB_NORM_TMP_OFF = UB_G_FP32_OFF;

// ============================================================
// AIC L1 buffer layout for Kernel 1
// ============================================================
// For compute_l_mqk_aic: need k_decayed, q_decayed, k_inv in L1
// For compute_neumann_inv_aic: need L, INV, Lpow, tmp in L1
//
// L1 offsets (bytes, aligned to 128)
constexpr int L1_KD_OFF    = 0;                          // [CHUNK, D] bf16 zN = 4096
constexpr int L1_QD_OFF    = L1_KD_OFF + 4096;           // [CHUNK, D] bf16 zN = 4096
constexpr int L1_KI_OFF    = L1_QD_OFF + 4096;           // [CHUNK, D] bf16 nZ = 4096
constexpr int L1_L_OFF     = L1_KI_OFF + 4096;           // [16, 16] fp16 zN = 512
constexpr int L1_INV_OFF   = L1_L_OFF + 512;             // [16, 16] fp16 zN = 512
constexpr int L1_LPOW_OFF  = L1_INV_OFF + 512;           // [16, 16] fp16 zN = 512
constexpr int L1_TMP_OFF   = L1_LPOW_OFF + 512;          // [16, 16] fp16 zN = 512
// Total L1: ~13KB

// ============================================================
// Kernel 1 class
// ============================================================
class FwdPrepareKernel {
public:
    __aicore__ FwdPrepareKernel() {}

    TPipe pipe;
    TBuf<PIPE_V> ubBuf;

    // AIC buffers
    TBuf<PIPE_MTE1> l1Buf;   // L1 buffer
    TBuf<PIPE_M>    l0ABuf;  // L0A buffer
    TBuf<PIPE_M>    l0BBuf;  // L0B buffer
    TBuf<PIPE_M>    l0CBuf;  // L0C buffer

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ void operator()(const FwdParams& params);

private:
    CrossCoreFlag elemReady{K1_ELEM_READY_ID};
    CrossCoreFlag mmaReady{K1_MMA_READY_ID};

    // Workspace index (set by AIC entry point for loading from GM)
    int ws_idx_ = 0;

    // ============================================================
    // AIV: Load inputs from GM to UB
    // ============================================================
    __aicore__ void load_inputs_aiv(
        const FwdParams& params, int head_idx,
        int64_t bos, int local_t
    ) {
        int64_t t_offset = bos + local_t * CHUNK;

        // Load q [CHUNK, D] bf16: GM → UB
        {
            LocalTensor<half> dst = ubBuf.GetBufferAddr<half>(UB_Q_OFF);
            GlobalTensor<half> src;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.q) + head_idx * params.T_total * D + t_offset * D,
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Load k [CHUNK, D] bf16
        {
            LocalTensor<half> dst = ubBuf.GetBufferAddr<half>(UB_K_OFF);
            GlobalTensor<half> src;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.k) + head_idx * params.T_total * D + t_offset * D,
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Load g [CHUNK, D] bf16
        {
            LocalTensor<half> dst = ubBuf.GetBufferAddr<half>(UB_G_BF16_OFF);
            GlobalTensor<half> src;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.g) + head_idx * params.T_total * D + t_offset * D,
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Load beta [CHUNK] bf16 (padded to 32 for alignment)
        {
            LocalTensor<half> dst = ubBuf.GetBufferAddr<half>(UB_BETA_OFF);
            GlobalTensor<half> src;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.beta) + head_idx * params.T_total + t_offset,
                CHUNK);
            DataCopy(dst, src, CHUNK);
        }

        // Load dt_bias [D] fp32 for this head
        {
            LocalTensor<float> dst = ubBuf.GetBufferAddr<float>(UB_DT_BIAS_OFF);
            GlobalTensor<float> src;
            src.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.dt_bias) + head_idx * D,
                D);
            DataCopy(dst, src, D);
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIV: L2 normalize q and k rows
    // ============================================================
    __aicore__ void l2_normalize_aiv() {
        LocalTensor<half> q_ub = ubBuf.GetBufferAddr<half>(UB_Q_OFF);
        LocalTensor<half> k_ub = ubBuf.GetBufferAddr<half>(UB_K_OFF);
        LocalTensor<float> sq_ub = ubBuf.GetBufferAddr<float>(UB_NORM_TMP_OFF);

        constexpr float eps = 1e-6f;

        for (int row = 0; row < CHUNK; ++row) {
            // --- Normalize q row ---
            LocalTensor<float> q_fp32 = sq_ub;
            Cast(q_fp32, q_ub[row * D], RoundMode::CAST_NONE, D);
            Mul(q_fp32, q_fp32, q_fp32, D);

            // ReduceSum: dst[0] = sum(src[0..D-1])
            LocalTensor<float> sum_dst = ubBuf.GetBufferAddr<float>(UB_NORM_TMP_OFF + D * sizeof(float));
            LocalTensor<float> reduce_tmp = ubBuf.GetBufferAddr<float>(UB_NORM_TMP_OFF + (D + 1) * sizeof(float));
            ReduceSum(sum_dst, q_fp32, reduce_tmp, D);
            float q_sq = sum_dst.GetValue(0);
            float q_inv = 1.0f / sqrtf(q_sq + eps);
            Muls(q_ub[row * D], q_ub[row * D], half(q_inv), D);

            // --- Normalize k row ---
            Cast(q_fp32, k_ub[row * D], RoundMode::CAST_NONE, D);
            Mul(q_fp32, q_fp32, q_fp32, D);
            ReduceSum(sum_dst, q_fp32, reduce_tmp, D);
            float k_sq = sum_dst.GetValue(0);
            float k_inv = 1.0f / sqrtf(k_sq + eps);
            Muls(k_ub[row * D], k_ub[row * D], half(k_inv), D);
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIV: Gate activation + cumulative sum
    // ============================================================
    __aicore__ void gate_activation_cumsum_aiv(
        const FwdParams& params, int head_idx, int actual_len
    ) {
        LocalTensor<half> g_bf16_ub = ubBuf.GetBufferAddr<half>(UB_G_BF16_OFF);
        LocalTensor<float> g_fp32_ub = ubBuf.GetBufferAddr<float>(UB_G_FP32_OFF);
        LocalTensor<float> dt_bias_ub = ubBuf.GetBufferAddr<float>(UB_DT_BIAS_OFF);
        LocalTensor<float> g_total_ub = ubBuf.GetBufferAddr<float>(UB_G_TOTAL_OFF);
        LocalTensor<half> k_ub = ubBuf.GetBufferAddr<half>(UB_K_OFF);

        // Read A_log for this head (scalar GM read)
        float a_log_val = 0.0f;
        {
            GlobalTensor<float> g_a_log;
            g_a_log.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.A_log) + head_idx, 1);
            LocalTensor<float> a_log_tmp = ubBuf.GetBufferAddr<float>(UB_G_TOTAL_OFF);
            DataCopy(a_log_tmp, g_a_log, 1);
            PipeBarrier<PIPE_V>();
            a_log_val = a_log_tmp.GetValue(0);
        }
        float a_log_exp = expf(a_log_val);
        float gate_scale = params.gate_scale;

        // Initialize g_total to zero
        Duplicate(g_total_ub, 0.0f, D);

        for (int row = 0; row < CHUNK; ++row) {
            // Cast g_bf16[row*D ..] to fp32
            Cast(g_fp32_ub[row * D], g_bf16_ub[row * D], RoundMode::CAST_NONE, D);

            if (row < actual_len) {
                // g_fp32 += dt_bias
                Add(g_fp32_ub[row * D], g_fp32_ub[row * D], dt_bias_ub, D);
                // g_fp32 *= a_log_exp
                Muls(g_fp32_ub[row * D], g_fp32_ub[row * D], a_log_exp, D);
                // sigmoid via tanh: sigmoid(x) = tanh(x/2) * 0.5 + 0.5
                Muls(g_fp32_ub[row * D], g_fp32_ub[row * D], 0.5f, D);
                Tanh(g_fp32_ub[row * D], g_fp32_ub[row * D], D);
                Muls(g_fp32_ub[row * D], g_fp32_ub[row * D], 0.5f, D);
                Adds(g_fp32_ub[row * D], g_fp32_ub[row * D], 0.5f, D);
                // g_fp32 *= gate_scale
                Muls(g_fp32_ub[row * D], g_fp32_ub[row * D], gate_scale, D);
            } else {
                Duplicate(g_fp32_ub[row * D], 0.0f, D);
            }

            // Cumulative sum: g_fp32[row*D + d] += g_total[d]
            Add(g_fp32_ub[row * D], g_fp32_ub[row * D], g_total_ub, D);
            // Update g_total = new running sum
            DataCopy(g_total_ub, g_fp32_ub[row * D], D);
        }

        // Zero k for tail rows
        if (actual_len < CHUNK) {
            for (int row = actual_len; row < CHUNK; ++row) {
                Duplicate(k_ub[row * D], half(0), D);
            }
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIV: Decay application
    // ============================================================
    __aicore__ void decay_apply_aiv(float scale) {
        LocalTensor<half> q_ub = ubBuf.GetBufferAddr<half>(UB_Q_OFF);
        LocalTensor<half> k_ub = ubBuf.GetBufferAddr<half>(UB_K_OFF);
        LocalTensor<float> g_fp32_ub = ubBuf.GetBufferAddr<float>(UB_G_FP32_OFF);
        LocalTensor<float> g_total_ub = ubBuf.GetBufferAddr<float>(UB_G_TOTAL_OFF);

        LocalTensor<half> q_decayed_ub = ubBuf.GetBufferAddr<half>(UB_Q_DECAYED_OFF);
        LocalTensor<half> k_decayed_ub = ubBuf.GetBufferAddr<half>(UB_K_DECAYED_OFF);
        LocalTensor<half> k_inv_ub = ubBuf.GetBufferAddr<half>(UB_K_INV_OFF);
        LocalTensor<half> k_restored_ub = ubBuf.GetBufferAddr<half>(UB_K_RESTORED_OFF);

        // Compute exp(g_total) once
        Exp(g_total_ub, g_total_ub, D);

        for (int row = 0; row < CHUNK; ++row) {
            int row_off = row * D;

            // exp_cumsum = exp(g_fp32[row, :])
            LocalTensor<float> exp_cumsum = g_fp32_ub[row_off];
            Exp(exp_cumsum, exp_cumsum, D);

            // Cast exp_cumsum to bf16
            LocalTensor<half> exp_bf16 = ubBuf.GetBufferAddr<half>(UB_G_FP32_OFF + row_off * sizeof(half));
            Cast(exp_bf16, exp_cumsum, RoundMode::CAST_RINT, D);

            // k_decayed = k * exp_cumsum
            Mul(k_decayed_ub[row_off], k_ub[row_off], exp_bf16, D);

            // q_decayed = q * exp_cumsum * scale
            Mul(q_decayed_ub[row_off], q_ub[row_off], exp_bf16, D);
            Muls(q_decayed_ub[row_off], q_decayed_ub[row_off], half(scale), D);

            // inv_cumsum = 1.0 / exp_cumsum
            LocalTensor<float> inv_cumsum_f = exp_cumsum;
            Duplicate(inv_cumsum_f, 1.0f, D);
            Div(inv_cumsum_f, inv_cumsum_f, exp_cumsum, D);

            // k_inv = k * inv_cumsum
            LocalTensor<half> inv_bf16 = ubBuf.GetBufferAddr<half>(UB_G_FP32_OFF + row_off * sizeof(half));
            Cast(inv_bf16, inv_cumsum_f, RoundMode::CAST_RINT, D);
            Mul(k_inv_ub[row_off], k_ub[row_off], inv_bf16, D);

            // k_restored = k_inv * exp(g_total)
            LocalTensor<half> exp_gt_bf16 = ubBuf.GetBufferAddr<half>(UB_G_FP32_OFF + row_off * sizeof(half));
            Cast(exp_gt_bf16, g_total_ub, RoundMode::CAST_RINT, D);
            Mul(k_restored_ub[row_off], k_inv_ub[row_off], exp_gt_bf16, D);
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIV: Lower-triangular mask + beta sigmoid + (I-L) construction
    // ============================================================
    __aicore__ void tril_mask_aiv(
        const FwdParams& params, int head_idx, int64_t t_offset
    ) {
        LocalTensor<half> L_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);
        LocalTensor<half> INV_ub = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
        LocalTensor<half> Mqk_ub = ubBuf.GetBufferAddr<half>(UB_MQK_OFF);
        LocalTensor<half> beta_ub = ubBuf.GetBufferAddr<half>(UB_BETA_OFF);

        for (int i = 0; i < CHUNK; ++i) {
            half beta_val = beta_ub.GetValue(i);
            float beta_f = static_cast<float>(beta_val);
            float sig_beta = sigmoid_tanh_approx(beta_f);
            half sig_beta_h = half(sig_beta);

            for (int j = 0; j < CHUNK; ++j) {
                int idx = i * CHUNK + j;
                half l_val = L_ub.GetValue(idx);

                if (i <= j) {
                    L_ub.SetValue(idx, half(0));
                    INV_ub.SetValue(idx, (i == j) ? half(1.0f) : half(0));
                } else {
                    half masked = half(static_cast<float>(l_val) * sig_beta);
                    L_ub.SetValue(idx, masked);
                    INV_ub.SetValue(idx, half(-static_cast<float>(masked)));
                }

                if (i < j) {
                    Mqk_ub.SetValue(idx, half(0));
                }
            }
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIV: Store workspace arrays to GM
    // ============================================================
    __aicore__ void store_workspace_aiv(const FwdParams& params, int ws_idx) {
        int64_t ws_base = (int64_t)ws_idx * WorkspaceSizes::kPerTile;

        // Store k_decayed [CHUNK, D] bf16
        {
            LocalTensor<half> src = ubBuf.GetBufferAddr<half>(UB_K_DECAYED_OFF);
            GlobalTensor<half> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKDecayed),
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Store q_decayed [CHUNK, D] bf16
        {
            LocalTensor<half> src = ubBuf.GetBufferAddr<half>(UB_Q_DECAYED_OFF);
            GlobalTensor<half> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kQDecayed),
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Store k_inv [CHUNK, D] bf16
        {
            LocalTensor<half> src = ubBuf.GetBufferAddr<half>(UB_K_INV_OFF);
            GlobalTensor<half> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKInv),
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Store k_restored [CHUNK, D] bf16
        {
            LocalTensor<half> src = ubBuf.GetBufferAddr<half>(UB_K_RESTORED_OFF);
            GlobalTensor<half> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKRestored),
                CHUNK * D);
            DataCopy(dst, src, CHUNK * D);
        }

        // Store g_total [D] fp32
        {
            LocalTensor<float> src = ubBuf.GetBufferAddr<float>(UB_G_TOTAL_OFF);
            GlobalTensor<float> dst;
            dst.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(params.workspace + ws_base + WorkspaceOffsets::kGTotal),
                D);
            DataCopy(dst, src, D);
        }

        PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // AIC: Compute L = k_decayed @ k_inv^T and Mqk = q_decayed @ k_inv^T
    // ============================================================
    // k_decayed [16, 128] @ k_inv^T [128, 16] → L [16, 16]
    //   = sum_{k=0}^{7} k_decayed[16, 16k:16(k+1)] @ k_inv[16, 16k:16(k+1)]^T
    //   Each sub-product: 16x16x16 MMAD
    //
    // Same for Mqk with q_decayed as A operand.
    //
    // Data path: GM (workspace) → L1 (Nd2Nz) → L0A/L0B (LoadData) → MMAD → L0C → UB (Fixpipe)
    __aicore__ void compute_l_mqk_aic(const FwdParams& params) {
        int64_t ws_base = (int64_t)ws_idx_ * WorkspaceSizes::kPerTile;

        // Nd2NzParams for GM RowMajor [CHUNK, D] → L1 zN format
        // Used for A operands (k_decayed, q_decayed)
        Nd2NzParams nd2nzA;
        nd2nzA.ndNum = 1;
        nd2nzA.nValue = CHUNK;     // 16 rows
        nd2nzA.dValue = D;         // 128 cols
        nd2nzA.srcDValue = D;      // RowMajor stride = D
        nd2nzA.srcNdMatrixStride = 0;
        nd2nzA.dstNzNStride = D / 16;  // zN: N-stride in C0 blocks
        nd2nzA.dstNzC0Stride = 1;
        nd2nzA.dstNzMatrixStride = 0;

        // Nd2NzParams for B operand (k_inv) — nZ format
        Nd2NzParams nd2nzB;
        nd2nzB.ndNum = 1;
        nd2nzB.nValue = CHUNK;
        nd2nzB.dValue = D;
        nd2nzB.srcDValue = D;
        nd2nzB.srcNdMatrixStride = 0;
        nd2nzB.dstNzNStride = D / 16;
        nd2nzB.dstNzC0Stride = 1;
        nd2nzB.dstNzMatrixStride = 0;

        // Load k_decayed from workspace GM → L1 zN
        {
            LocalTensor<half> l1_dst = l1Buf.GetBufferAddr<half>(L1_KD_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKDecayed),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nzA);
        }

        // Load q_decayed from workspace GM → L1 zN
        {
            LocalTensor<half> l1_dst = l1Buf.GetBufferAddr<half>(L1_QD_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kQDecayed),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nzA);
        }

        // Load k_inv from workspace GM → L1 nZ
        {
            LocalTensor<half> l1_dst = l1Buf.GetBufferAddr<half>(L1_KI_OFF);
            GlobalTensor<half> gm_src;
            gm_src.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kKInv),
                CHUNK * D);
            DataCopy(l1_dst, gm_src, nd2nzB);
        }

        // Wait for GM → L1 DataCopy to complete
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        // LoadData params for L1 → L0A (A operand, no transpose)
        LoadData2DParams ldParamsA;
        ldParamsA.startIndex = 0;
        ldParamsA.repeatTimes = 1;   // 16 elements per repeat
        ldParamsA.srcStride = 1;
        ldParamsA.sid = 0;
        ldParamsA.dstGap = 0;
        ldParamsA.ifTranspose = false;
        ldParamsA.addrMode = 0;

        // LoadData params for L1 → L0B (B operand, transposed for k_inv^T)
        LoadData2DParams ldParamsB;
        ldParamsB.startIndex = 0;
        ldParamsB.repeatTimes = 1;
        ldParamsB.srcStride = 1;
        ldParamsB.sid = 0;
        ldParamsB.dstGap = 0;
        ldParamsB.ifTranspose = true;   // k_inv^T: transpose B
        ldParamsB.addrMode = 0;

        // ---- Compute L = k_decayed @ k_inv^T ----
        {
            LocalTensor<float> l0C = l0CBuf.GetBufferAddr<float>(0);

            for (int k = 0; k < D / 16; ++k) {
                // Load k_decayed[16, 16k:16(k+1)] from L1 → L0A
                // In zN format, the k-th D-block offset is k * FRACTAL_BLOCK_BYTES
                LocalTensor<half> l0A = l0ABuf.GetBufferAddr<half>(0);
                LocalTensor<half> l1A = l1Buf.GetBufferAddr<half>(
                    L1_KD_OFF + l1_zn_block_off(0, k, D));
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
                LoadData(l0A, l1A, ldParamsA);

                // Load k_inv[16, 16k:16(k+1)] from L1 → L0B (transposed)
                LocalTensor<half> l0B = l0BBuf.GetBufferAddr<half>(0);
                LocalTensor<half> l1B = l1Buf.GetBufferAddr<half>(
                    L1_KI_OFF + l1_zn_block_off(0, k, D));
                LoadData(l0B, l1B, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                // MMAD: l0C += l0A @ l0B
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = 16;
                mmadParams.n = 16;
                mmadParams.k = 16;
                mmadParams.cmatrixInitVal = (k == 0);  // init accumulator on first k-block
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            }

            // Fixpipe: L0C → UB (L result)
            // L0C is fp32 accumulator in zN format; convert to bf16 RowMajor in UB
            LocalTensor<half> L_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            FixpipeParamsV220 fixpipeParams;
            fixpipeParams.nSize = 16;
            fixpipeParams.mSize = 16;
            fixpipeParams.srcStride = 1;
            fixpipeParams.dstStride = CHUNK;  // RowMajor stride in UB
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
            fixpipeParams.reluEn = false;
            fixpipeParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(L_ub, l0C, fixpipeParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }

        // ---- Compute Mqk = q_decayed @ k_inv^T ----
        // Same structure, different A operand (q_decayed instead of k_decayed)
        {
            LocalTensor<float> l0C = l0CBuf.GetBufferAddr<float>(0);

            for (int k = 0; k < D / 16; ++k) {
                LocalTensor<half> l0A = l0ABuf.GetBufferAddr<half>(0);
                LocalTensor<half> l1A = l1Buf.GetBufferAddr<half>(
                    L1_QD_OFF + l1_zn_block_off(0, k, D));
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
                LoadData(l0A, l1A, ldParamsA);

                LocalTensor<half> l0B = l0BBuf.GetBufferAddr<half>(0);
                LocalTensor<half> l1B = l1Buf.GetBufferAddr<half>(
                    L1_KI_OFF + l1_zn_block_off(0, k, D));
                LoadData(l0B, l1B, ldParamsB);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                MmadParams mmadParams;
                mmadParams.m = 16;
                mmadParams.n = 16;
                mmadParams.k = 16;
                mmadParams.cmatrixInitVal = (k == 0);
                mmadParams.unitFlag = 0;
                Mmad(l0C, l0A, l0B, mmadParams);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            }

            // Fixpipe: L0C → UB (Mqk result)
            LocalTensor<half> Mqk_ub = ubBuf.GetBufferAddr<half>(UB_MQK_OFF);
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            FixpipeParamsV220 fixpipeParams;
            fixpipeParams.nSize = 16;
            fixpipeParams.mSize = 16;
            fixpipeParams.srcStride = 1;
            fixpipeParams.dstStride = CHUNK;
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
            fixpipeParams.reluEn = false;
            fixpipeParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(Mqk_ub, l0C, fixpipeParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    // ============================================================
    // AIC: Neumann inverse INV = (I - L)^{-1}
    // ============================================================
    // After tril_mask_aiv, INV_ub contains (I - L) and L_ub contains L.
    // Neumann series:
    //   L^2 = L × L
    //   INV = (I-L) + (I-L) × L^2
    //   L^4 = L^2 × L^2
    //   INV += INV × L^4
    //   L^8 = L^4 × L^4
    //   INV += INV × L^8
    //
    // Each 16x16 MMAD: LoadData L1→L0A/L0B, Mmad, Fixpipe L0C→L1
    // Between MMADs: copy L0C result to L1 for next iteration
    __aicore__ void compute_neumann_inv_aic() {
        // Copy L and (I-L) from UB → L1
        // L is in UB_L_OFF as bf16 [16, 16]
        // (I-L) is in UB_INV_OFF as bf16 [16, 16] (constructed by tril_mask_aiv)
        // For MMAD, we need fp16 format in L1 zN

        // UB → L1 copy (simple DataCopy, no fractal conversion for small 16x16)
        // Note: For 16x16 matrices, the zN format in L1 is the same as RowMajor
        // since 16 is one fractal block.
        {
            LocalTensor<half> L_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);
            LocalTensor<half> L_l1 = l1Buf.GetBufferAddr<half>(L1_L_OFF);
            DataCopy(L_l1, L_ub, CHUNK * CHUNK);
        }
        {
            LocalTensor<half> INV_ub = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
            LocalTensor<half> INV_l1 = l1Buf.GetBufferAddr<half>(L1_INV_OFF);
            DataCopy(INV_l1, INV_ub, CHUNK * CHUNK);
        }

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

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
        ldParamsB.ifTranspose = true;  // transpose for B operand
        ldParamsB.addrMode = 0;

        MmadParams mmadParams;
        mmadParams.m = 16;
        mmadParams.n = 16;
        mmadParams.k = 16;
        mmadParams.unitFlag = 0;

        // Helper lambda: MMAD C = A @ B^T, result stored to L1 dst_off
        // Then copy L1 dst to L1 for next use
        auto mmad_16x16 = [&](int l1_a_off, int l1_b_off, int l1_c_off, bool init) {
            LocalTensor<half> l0A = l0ABuf.GetBufferAddr<half>(0);
            LocalTensor<half> l0B = l0BBuf.GetBufferAddr<half>(0);
            LocalTensor<float> l0C = l0CBuf.GetBufferAddr<float>(0);
            LocalTensor<half> l1A = l1Buf.GetBufferAddr<half>(l1_a_off);
            LocalTensor<half> l1B = l1Buf.GetBufferAddr<half>(l1_b_off);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            LoadData(l0A, l1A, ldParamsA);
            LoadData(l0B, l1B, ldParamsB);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            mmadParams.cmatrixInitVal = init ? 1 : 0;
            Mmad(l0C, l0A, l0B, mmadParams);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);

            // Fixpipe: L0C → L1
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            LocalTensor<half> l1C = l1Buf.GetBufferAddr<half>(l1_c_off);
            FixpipeParamsV220 fixParams;
            fixParams.nSize = 16;
            fixParams.mSize = 16;
            fixParams.srcStride = 1;
            fixParams.dstStride = CHUNK;
            fixParams.quantPre = QuantMode_t::F322BF16;
            fixParams.reluEn = false;
            fixParams.unitFlag = 0;
            Fixpipe<half, float, CFG_ROW_MAJOR>(l1C, l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        };

        // Step 1: L^2 = L × L
        mmad_16x16(L1_L_OFF, L1_L_OFF, L1_LPOW_OFF, true);

        // Step 2: INV = (I-L) + (I-L) × L^2
        // tmp = (I-L) × L^2
        mmad_16x16(L1_INV_OFF, L1_LPOW_OFF, L1_TMP_OFF, true);
        // INV += tmp (element-wise add in L1)
        // On AIC, element-wise add on L1 requires UB intermediate:
        //   Copy INV and tmp to UB, Add, Copy back
        // Or: use AIV for the add (requires CrossCore sync — expensive)
        // Alternative: fuse the add into the next MMAD by using bias
        // For simplicity, do the add via UB:
        {
            LocalTensor<half> inv_l1 = l1Buf.GetBufferAddr<half>(L1_INV_OFF);
            LocalTensor<half> tmp_l1 = l1Buf.GetBufferAddr<half>(L1_TMP_OFF);
            LocalTensor<half> inv_ub = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
            LocalTensor<half> tmp_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);  // reuse L UB space
            DataCopy(inv_ub, inv_l1, CHUNK * CHUNK);
            DataCopy(tmp_ub, tmp_l1, CHUNK * CHUNK);
            PipeBarrier<PIPE_V>();
            Add(inv_ub, inv_ub, tmp_ub, CHUNK * CHUNK);
            DataCopy(inv_l1, inv_ub, CHUNK * CHUNK);
            PipeBarrier<PIPE_V>();
        }

        // Step 3: L^4 = L^2 × L^2
        mmad_16x16(L1_LPOW_OFF, L1_LPOW_OFF, L1_LPOW_OFF, true);

        // Step 4: INV += INV × L^4
        mmad_16x16(L1_INV_OFF, L1_LPOW_OFF, L1_TMP_OFF, true);
        {
            LocalTensor<half> inv_l1 = l1Buf.GetBufferAddr<half>(L1_INV_OFF);
            LocalTensor<half> tmp_l1 = l1Buf.GetBufferAddr<half>(L1_TMP_OFF);
            LocalTensor<half> inv_ub = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
            LocalTensor<half> tmp_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);
            DataCopy(inv_ub, inv_l1, CHUNK * CHUNK);
            DataCopy(tmp_ub, tmp_l1, CHUNK * CHUNK);
            PipeBarrier<PIPE_V>();
            Add(inv_ub, inv_ub, tmp_ub, CHUNK * CHUNK);
            DataCopy(inv_l1, inv_ub, CHUNK * CHUNK);
            PipeBarrier<PIPE_V>();
        }

        // Step 5: L^8 = L^4 × L^4
        mmad_16x16(L1_LPOW_OFF, L1_LPOW_OFF, L1_LPOW_OFF, true);

        // Step 6: INV += INV × L^8
        mmad_16x16(L1_INV_OFF, L1_LPOW_OFF, L1_TMP_OFF, true);
        {
            LocalTensor<half> inv_l1 = l1Buf.GetBufferAddr<half>(L1_INV_OFF);
            LocalTensor<half> tmp_l1 = l1Buf.GetBufferAddr<half>(L1_TMP_OFF);
            LocalTensor<half> inv_ub = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
            LocalTensor<half> tmp_ub = ubBuf.GetBufferAddr<half>(UB_L_OFF);
            DataCopy(inv_ub, inv_l1, CHUNK * CHUNK);
            DataCopy(tmp_ub, tmp_l1, CHUNK * CHUNK);
            PipeBarrier<PIPE_V>();
            Add(inv_ub, inv_ub, tmp_ub, CHUNK * CHUNK);
            // Final INV result stays in UB for AIV to store
            // No need to copy back to L1
        }
    }
};

// ============================================================
// AIV entry point
// ============================================================
template <>
__aicore__ void FwdPrepareKernel::operator()<AIV>(const FwdParams& params) {
    pipe.InitBuffer(ubBuf, PIPE_V, 192 * 1024);

    int linear_idx = GetBlockIdx();
    int head_idx = linear_idx / params.total_tiles;
    int global_tile_idx = linear_idx % params.total_tiles;

    // Compute sequence info
    int seq_idx = 0, local_t = 0;
    int64_t bos = 0, eos = 0;
    int tiles_before = 0;

    if (params.is_varlen) {
        for (int i = 0; i < params.N; i++) {
            int64_t seq_start = 0, seq_end = 0;
            {
                GlobalTensor<int64_t> g_cu;
                g_cu.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i, 1);
                LocalTensor<int64_t> cu_tmp = ubBuf.GetBufferAddr<int64_t>(
                    UB_MQK_OFF + UB_LM_SIZE);
                DataCopy(cu_tmp, g_cu, 1);
                PipeBarrier<PIPE_V>();
                seq_start = cu_tmp.GetValue(0);

                g_cu.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int64_t*>(params.cu_seqlens) + i + 1, 1);
                DataCopy(cu_tmp, g_cu, 1);
                PipeBarrier<PIPE_V>();
                seq_end = cu_tmp.GetValue(0);
            }

            int slen = int(seq_end - seq_start);
            int n_tiles = (slen + CHUNK - 1) / CHUNK;
            if (tiles_before + n_tiles > global_tile_idx) {
                seq_idx = i;
                bos = seq_start;
                eos = seq_end;
                break;
            }
            tiles_before += n_tiles;
        }
        local_t = global_tile_idx - tiles_before;
    } else {
        int T_seq = params.T_total / params.N;
        int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = global_tile_idx / tiles_per_seq;
        tiles_before = seq_idx * tiles_per_seq;
        local_t = global_tile_idx - tiles_before;
        bos = (int64_t)seq_idx * T_seq;
        eos = bos + T_seq;
    }

    int seq_len = int(eos - bos);
    int t_tiles_this_seq = (seq_len + CHUNK - 1) / CHUNK;
    if (local_t >= t_tiles_this_seq) return;
    int actual_len = min(CHUNK, seq_len - local_t * CHUNK);

    // Phase 1: Load inputs
    load_inputs_aiv(params, head_idx, bos, local_t);

    // Phase 2: L2 normalize q and k
    l2_normalize_aiv();

    // Phase 3: Gate activation + cumulative sum
    gate_activation_cumsum_aiv(params, head_idx, actual_len);

    // Phase 4: Decay application
    decay_apply_aiv(params.scale);

    // Phase 5: Store workspace (k_decayed, q_decayed, k_restored, g_total)
    int ws_idx = head_idx * params.total_tiles + global_tile_idx;
    store_workspace_aiv(params, ws_idx);

    // Signal AIC: element-wise done, workspace in GM
    CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady);

    // Wait for AIC: L and Mqk computed
    CrossCoreWaitFlag(mmaReady);

    // Phase 6: Apply tril mask + beta sigmoid + construct (I-L)
    int64_t t_offset = bos + local_t * CHUNK;
    tril_mask_aiv(params, head_idx, t_offset);

    // Signal AIC: (I-L) ready for Neumann inverse
    CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady);

    // Wait for AIC: INV computed
    CrossCoreWaitFlag(mmaReady);

    // Phase 8: Store INV and Mqk to workspace
    {
        int64_t ws_base = (int64_t)ws_idx * WorkspaceSizes::kPerTile;

        LocalTensor<half> inv_src = ubBuf.GetBufferAddr<half>(UB_INV_OFF);
        GlobalTensor<half> inv_dst;
        inv_dst.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kINV),
            CHUNK * CHUNK);
        DataCopy(inv_dst, inv_src, CHUNK * CHUNK);

        LocalTensor<half> mqk_src = ubBuf.GetBufferAddr<half>(UB_MQK_OFF);
        GlobalTensor<half> mqk_dst;
        mqk_dst.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(params.workspace + ws_base + WorkspaceOffsets::kMqk),
            CHUNK * CHUNK);
        DataCopy(mqk_dst, mqk_src, CHUNK * CHUNK);

        PipeBarrier<PIPE_V>();
    }
}

// ============================================================
// AIC entry point
// ============================================================
template <>
__aicore__ void FwdPrepareKernel::operator()<AIC>(const FwdParams& params) {
    // Initialize buffers
    pipe.InitBuffer(l1Buf, PIPE_MTE1, L1_SIZE);
    pipe.InitBuffer(l0ABuf, PIPE_M, L0A_SIZE);
    pipe.InitBuffer(l0BBuf, PIPE_M, L0B_SIZE);
    pipe.InitBuffer(l0CBuf, PIPE_M, L0C_SIZE);

    // Set workspace index (same as AIV)
    int linear_idx = GetBlockIdx();
    int head_idx = linear_idx / params.total_tiles;
    int global_tile_idx = linear_idx % params.total_tiles;
    ws_idx_ = head_idx * params.total_tiles + global_tile_idx;

    // Initialize HardEvent flags
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);

    // Wait for AIV: element-wise done, workspace in GM
    CrossCoreWaitFlag(elemReady);

    // Phase 6: Compute L and Mqk via MMAD
    compute_l_mqk_aic(params);

    // Signal AIV: L and Mqk ready in UB
    CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady);

    // Wait for AIV: (I-L) constructed in UB
    CrossCoreWaitFlag(elemReady);

    // Phase 7: Neumann inverse
    compute_neumann_inv_aic();

    // Signal AIV: INV ready in UB
    CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady);

    // Release HardEvent flags
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

}  // namespace flash_kda
