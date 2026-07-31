#pragma once

#include <acl/acl.h>

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

// Diagnostic: smallest kernel on this launch path.
void launch_noop(GM_ADDR flag, aclrtStream stream);

// Diagnostic: handshake only, real FwdParams by value.
void launch_sync_only(const FwdParams& params, aclrtStream stream);

// Diagnostic: same handshake, small scalar args.
void launch_sync_small(int units, aclrtStream stream);

// Diagnostic: AIV-only, no cross-core sync.
void launch_aiv_only(GM_ADDR src, GM_ADDR dst, aclrtStream stream);
void launch_aic_only(GM_ADDR flag, aclrtStream stream);
void launch_sync_onearg(const FwdParams& params, aclrtStream stream);

}  // namespace flash_kda
