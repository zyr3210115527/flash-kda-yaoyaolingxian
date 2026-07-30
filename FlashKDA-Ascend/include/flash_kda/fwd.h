#pragma once

#include "flash_kda/layout.hpp"

namespace flash_kda {

// ============================================================
// Kernel launch declarations
// ============================================================

// Kernel 1: Prepare
// Grid: (total_tiles * H) — one block per (tile, head)
// AIV-dominant: element-wise ops (normalize, gate, cumsum, decay)
// AIC: 3x 16x16 MMA (L, Mqk, Neumann inverse)
//
// Computes and stores to workspace:
//   k_decayed [CHUNK, D] bf16
//   q_decayed [CHUNK, D] bf16
//   k_restored [CHUNK, D] bf16
//   g_total [D] float
//   INV [CHUNK, CHUNK] bf16
//   Mqk [CHUNK, CHUNK] bf16
void launch_fwd_prepare(const FwdParams& params, aclrtStream stream);

// Kernel 2: Recurrence
// Grid: (N * H) — one block per (sequence, head)
// AIC-dominant: MMAD pipeline (dual GEMM, INV@u, state update)
// AIV: element-wise ops (beta sigmoid, state * exp(g_total))
//
// Reads workspace from Kernel 1, computes:
//   out = q_decayed @ state + Mqk @ (INV @ ((v - k_decayed @ state) * beta))
//   state = state * exp(g_total) + k_restored^T @ u
void launch_fwd_recurrence(const FwdParams& params, aclrtStream stream);

}  // namespace flash_kda
