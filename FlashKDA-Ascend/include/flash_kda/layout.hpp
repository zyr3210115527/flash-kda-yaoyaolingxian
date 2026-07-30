#pragma once

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"

namespace flash_kda {

// ============================================================
// Compile-time constants
// ============================================================

constexpr int CHUNK = 16;
constexpr int D = 128;  // K = V = 128

// Ascend architecture tag. CATLASS_ARCH selects the SoC generation:
//   2201 -> Atlas A2 / A3 (Ascend 910B / 910C)
//   3510 -> Ascend 950
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using ArchTag = Catlass::Arch::Ascend950;
#else
using ArchTag = Catlass::Arch::AtlasA2;
#endif

// Memory sizes. Names must match catlass/arch/arch.hpp exactly.
constexpr uint32_t UB_SIZE  = ArchTag::UB_SIZE;
constexpr uint32_t L1_SIZE  = ArchTag::L1_SIZE;
constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;

// Fractal block: 16x16 elements. For 2-byte types one fractal is 512 bytes.
constexpr int C0_NUM_PER_FRACTAL = 16;

// ============================================================
// Data types
// ============================================================
//
// The inherited draft used `ascendFloat16`, which does not exist in AscendC,
// and claimed "AscendC uses half for bf16 on A2". Both are wrong: bf16 is
// `bfloat16_t`, and Atlas A2 supports it natively for both MMAD and the
// Fixpipe F322BF16 quantization mode. half and bfloat16_t have different
// exponent widths (5 vs 8), so reinterpreting one as the other yields garbage,
// not merely reduced precision.

using BF16 = bfloat16_t;
using FP16 = half;
using FP32 = float;

// ============================================================
// Workspace layout
// ============================================================
//
// RowMajor in GM so it interoperates with PyTorch tensors. Fractal (zN/nZ)
// layouts only exist inside L1/L0.
//
// On Atlas A2 the AIC has no access to UB, and there is no L0C->UB path
// (catlass gemm/tile/copy_l0c_to_ub.hpp is guarded by CATLASS_ARCH == 3510).
// Every AIC result therefore lands in GM via Fixpipe, and the AIV reads it
// back from GM. The scratch slots below exist for that hand-off.

struct WorkspaceSizes {
    static constexpr int kKDecayed  = CHUNK * D * sizeof(BF16);   // 4096
    static constexpr int kQDecayed  = CHUNK * D * sizeof(BF16);   // 4096
    static constexpr int kKInv      = CHUNK * D * sizeof(BF16);   // 4096
    static constexpr int kKRestored = CHUNK * D * sizeof(BF16);   // 4096
    static constexpr int kGTotal    = D * sizeof(FP32);           // 512
    static constexpr int kINV       = CHUNK * CHUNK * sizeof(BF16);  // 512
    static constexpr int kMqk       = CHUNK * CHUNK * sizeof(BF16);  // 512

    // A 16x16 bf16 identity. The Neumann iteration applies P(I + X) as
    // P + P*X, seeding L0C with one MMAD of P against this identity. Keeping
    // it in the workspace is what lets the whole inverse stay on the cube --
    // the AIC has no vector unit to add with.
    static constexpr int kIdentity  = CHUNK * CHUNK * sizeof(BF16);  // 512

    // AIC <-> AIV scratch for the L / Neumann hand-off. Six 16x16 slots:
    // L, (I-L), L^2, L^4, L^8, and one spare accumulator.
    // Kernel 2 stages every AIC<->AIV handoff through these slots, because A2
    // has no L1<->UB path. The widest thing that crosses is the [D, D] state
    // update in fp32, so every slot is sized for that.
    static constexpr int kScratchSlot = D * D * sizeof(FP32);          // 65536
    static constexpr int kNumScratch  = 9;
    static constexpr int kScratch     = kScratchSlot * kNumScratch;

    static constexpr int64_t kPerTile =
        kKDecayed + kQDecayed + kKInv + kKRestored + kGTotal + kINV + kMqk + kIdentity + kScratch;
};

struct WorkspaceOffsets {
    static constexpr int kKDecayed  = 0;
    static constexpr int kQDecayed  = kKDecayed + WorkspaceSizes::kKDecayed;
    static constexpr int kKInv      = kQDecayed + WorkspaceSizes::kQDecayed;
    static constexpr int kKRestored = kKInv + WorkspaceSizes::kKInv;
    static constexpr int kGTotal    = kKRestored + WorkspaceSizes::kKRestored;
    static constexpr int kINV       = kGTotal + WorkspaceSizes::kGTotal;
    static constexpr int kMqk       = kINV + WorkspaceSizes::kINV;
    static constexpr int kIdentity  = kMqk + WorkspaceSizes::kMqk;
    static constexpr int kScratch   = kIdentity + WorkspaceSizes::kIdentity;
};

// ============================================================
// Kernel parameters
// ============================================================

struct FwdParams {
    // Inputs (GM)
    GM_ADDR q;         // [T_total, H, D] bf16
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR g;
    GM_ADDR beta;      // [H, T_total] bf16 (transposed host-side)
    GM_ADDR A_log;     // [H] fp32
    GM_ADDR dt_bias;   // [H, D] fp32

    // Output
    GM_ADDR out;       // [T_total, H, D] bf16

    GM_ADDR workspace;

    // Optional state, nullptr when absent
    GM_ADDR initial_state;  // [N, H, D, D]
    GM_ADDR final_state;

    GM_ADDR cu_seqlens;     // [N+1] int64, nullptr when not varlen

    // Byte offset into `workspace` of the live [N, H, D, D] fp32 recurrent
    // state that kernel 2 carries across chunks. Sits past the per-tile region.
    int64_t state_ws_offset;

    // Scalars
    float scale;

    // Gate lower bound. The CUDA kernel folds log2(e) in here because it uses
    // ex2 (2^x); this port uses the natural Exp, so gate_scale is the raw
    // lower_bound and the two agree exactly:
    //   2^(lower_bound * log2e * s) == e^(lower_bound * s)
    float gate_scale;

    // Dimensions
    int T_total;
    int H;
    int N;
    int total_tiles;

    // Feature flags (runtime, not template specialization)
    int has_state_in;
    int has_state_out;
    int state_fp32;
    int is_varlen;
};

}  // namespace flash_kda
