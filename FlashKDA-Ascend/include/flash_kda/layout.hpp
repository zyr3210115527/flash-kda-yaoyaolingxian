#pragma once

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"

namespace flash_kda {

// Compile-time constants
constexpr int CHUNK = 16;
constexpr int D = 128;  // K = V = 128

// Ascend architecture tag
using ArchTag = Catlass::Arch::AtlasA2;

// Memory sizes from architecture
constexpr int UB_SIZE = ArchTag::UBSize;      // 192KB
constexpr int L1_SIZE = ArchTag::L1Size;      // 512KB
constexpr int L0A_SIZE = ArchTag::L0ASize;    // 64KB
constexpr int L0B_SIZE = ArchTag::L0BSize;    // 64KB
constexpr int L0C_SIZE = ArchTag::L0CSize;    // 128KB

// Fractal block size: 16 elements per fractal inner dimension
constexpr int C0_NUM_PER_FRACTAL = 16;

// Byte size of one fractal block [C0, C0] in L1 zN/nZ format
// Used for computing L1 offsets when indexing into fractal-layout data
constexpr int FRACTAL_BLOCK_BYTES = C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL * sizeof(half);  // 512

// Helper: L1 zN offset for block [n_blk, d_blk] of a [rows, cols] matrix
// rows/cols must be multiples of C0_NUM_PER_FRACTAL
inline constexpr int l1_zn_block_off(int n_blk, int d_blk, int cols) {
    return (n_blk * (cols / C0_NUM_PER_FRACTAL) + d_blk) * FRACTAL_BLOCK_BYTES;
}

// Data type aliases matching AscendC types
using BF16 = ascendFloat16;  // Note: AscendC uses half for bf16 on A2
using FP16 = half;
using FP32 = float;

// Workspace layout: RowMajor in GM (compatible with PyTorch tensors)
// Internal layouts: Fractal (zN/nZ) in L1/L0

// Workspace per-tile byte sizes (same as CUDA version)
struct WorkspaceSizes {
    static constexpr int kKDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kQDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kKInv      = CHUNK * D * 2;        // 4096
    static constexpr int kKRestored = CHUNK * D * 2;        // 4096
    static constexpr int kGTotal    = D * 4;                 // 512
    static constexpr int kINV       = CHUNK * CHUNK * 2;     // 512
    static constexpr int kMqk       = CHUNK * CHUNK * 2;     // 512
    static constexpr int64_t kPerTile = kKDecayed + kQDecayed + kKInv + kKRestored + kGTotal + kINV + kMqk;
};

// Workspace offsets within the per-tile buffer
struct WorkspaceOffsets {
    static constexpr int kKDecayed  = 0;
    static constexpr int kQDecayed  = kKDecayed + WorkspaceSizes::kKDecayed;
    static constexpr int kKInv      = kQDecayed + WorkspaceSizes::kQDecayed;
    static constexpr int kKRestored = kKInv + WorkspaceSizes::kKInv;
    static constexpr int kGTotal    = kKRestored + WorkspaceSizes::kKRestored;
    static constexpr int kINV       = kGTotal + WorkspaceSizes::kGTotal;
    static constexpr int kMqk       = kINV + WorkspaceSizes::kINV;
};

// Kernel parameters passed from host
struct FwdParams {
    // Input pointers (GM)
    __gm__ BF16* q;           // [B, T, H, D] or [1, T_total, H, D]
    __gm__ BF16* k;
    __gm__ BF16* v;
    __gm__ BF16* g;
    __gm__ BF16* beta;        // [B, T, H]
    __gm__ FP32* A_log;       // [H]
    __gm__ FP32* dt_bias;     // [H, D]

    // Output pointer
    __gm__ BF16* out;         // [B, T, H, D]

    // Workspace pointer
    __gm__ uint8_t* workspace;

    // State pointers (may be nullptr)
    __gm__ void* initial_state;  // [N, H, D, D] bf16 or fp32
    __gm__ void* final_state;

    // Varlen
    __gm__ int64_t* cu_seqlens;  // [N+1], may be nullptr

    // Scalars
    float scale;
    float gate_scale;  // lower_bound * 1.4426950408889634

    // Dimensions
    int T_total;
    int H;
    int N;
    int total_tiles;

    // Flags (packed as int for easy host-side construction)
    int has_state_in;
    int has_state_out;
    int state_fp32;
    int is_varlen;
};

}  // namespace flash_kda
