#pragma once

#include "flash_kda/layout.hpp"

namespace flash_kda {

// ============================================================
// Device-side math helpers
// ============================================================
//
// Everything callable from a kernel must be marked __aicore__, otherwise the
// device compiler rejects it as "calling a host function from device code".
// The inherited draft declared these plain `inline` and called them from
// __aicore__ code, which cannot compile.
//
// Note on exp bases: the CUDA kernel uses ex2 (2^x) and folds log2(e) into
// gate_scale on the host. This port uses the natural Exp everywhere and passes
// the raw lower_bound as gate_scale, which is exactly equivalent:
//   2^(lower_bound * log2e * s) == e^(lower_bound * s)
// Mixing the two conventions -- ex2's gate_scale with a natural Exp -- inflates
// every decay exponent by log2(e) ~= 1.4427, which is the single largest
// numerical error in the inherited draft.

constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2   = 0.6931471805599453f;

// There is no scalar exp intrinsic on the aicore -- only `sqrt` and friends
// exist in __clang_cce_aicore_functions.h. Anything exponential has to go
// through the vector unit, so a "scalar" exp means staging the value in UB,
// running AscendC::Exp over one datablock, and reading it back. Callers that
// need this use ExpViaVector in the kernel; there is deliberately no scalar
// helper here that would look cheap and silently fail to compile.

// ============================================================
// Fractal (zN) addressing
// ============================================================
//
// zN is row-major *inside* a 16x16 fractal and column-major *between*
// fractals (catlass/include/catlass/layout/matrix.hpp). The inherited draft
// had the two block indices swapped, which happened to be harmless only
// because every matrix it addressed had a single fractal row (rows == 16).
// It breaks the moment a matrix is taller than one fractal -- which is the
// case for the [128, 128] recurrent state.
//
// Returns a byte offset for element type of size `elem_bytes`.
__aicore__ inline int ZnBlockOffsetBytes(int row_blk, int col_blk, int rows, int elem_bytes)
{
    const int rows_round = (rows + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL;
    const int fractal_bytes = C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL * elem_bytes;
    return (col_blk * (rows_round / C0_NUM_PER_FRACTAL) + row_blk) * fractal_bytes;
}

// Nd2Nz destination strides for a [rows, cols] RowMajor GM source landing in
// L1 as zN. Derived from catlass gemm/tile/atlasa2/copy_gm_to_l1.hpp:
//   dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0 = roundUp16(rows)
//   dstNzNStride  = layoutDst.stride(0) / ELE_NUM_PER_C0 = 1
// The draft had these two swapped.
__aicore__ inline uint16_t Nd2NzC0Stride(int rows)
{
    return static_cast<uint16_t>((rows + C0_NUM_PER_FRACTAL - 1) / C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL);
}

__aicore__ inline uint16_t Nd2NzNStride()
{
    return 1;
}

}  // namespace flash_kda
