#pragma once

#include "kernel_operator.h"

namespace flash_kda {

// ============================================================
// Math approximations matching FlashKDA CUDA kernel behavior
// ============================================================

// Sigmoid via tanh approximation: sigmoid(x) = tanh(x/2) / 2 + 0.5
// Matches the CUDA asm "tanh.approx.f32" used in the original kernel
inline float sigmoid_tanh_approx(float x) {
    float th;
    // AscendC vector tanh — on AIV use AscendC::Tanh
    // For scalar use in host/tiling code:
    th = tanhf(x * 0.5f);
    return th * 0.5f + 0.5f;
}

// exp2 approximation matching CUDA "ex2.approx.ftz.f32"
// On Ascend AIV, use AscendC::Exp which computes e^x
// For scalar: use expf()
inline float ex2_approx(float x) {
    // CUDA ex2.approx computes 2^x
    // AscendC::Exp computes e^x
    // Convert: 2^x = e^(x * ln2)
    return expf(x * 0.6931471805599453f);
}

// BF16 <-> FP32 conversion
// On AIV, use AscendC::Cast for vectorized conversion
// For scalar:
inline float bf16_to_f32(uint16_t bf16_bits) {
    // BF16 = FP32 with lower 16 bits zeroed
    uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float x) {
    // Round-to-nearest-even truncation
    uint32_t f32_bits;
    memcpy(&f32_bits, &x, sizeof(uint32_t));
    // Add rounding bias
    f32_bits += 0x8000;
    // Truncate lower 16 bits
    return static_cast<uint16_t>(f32_bits >> 16);
}

// ============================================================
// Workspace size computation (host-side, matches CUDA version)
// ============================================================

inline int64_t get_workspace_size(int64_t T_total, int64_t H, int64_t N = 1) {
    constexpr int CHUNK = 16;
    constexpr int D = 128;

    // Upper bound: each of N sequences adds at most 1 extra tile vs floor division
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

    int64_t per_tile_bytes = 4 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;

    return H * total_tiles * per_tile_bytes;
}

}  // namespace flash_kda
