/**
 * FlashKDA Ascend — Kernel 2 Launch (Recurrence)
 *
 * Host-side kernel launch for FwdRecurrenceKernel.
 * Grid: (N * H) — one AIC+AIV core per (sequence, head)
 */

#include "flash_kda/fwd.h"
#include "flash_kda/fwd_kernel2.hpp"

namespace flash_kda {

void launch_fwd_recurrence(const FwdParams& params, aclrtStream stream) {
    // Grid dimension: N * H blocks
    // Each block = 1 AIC core + 1 AIV core
    int block_dim = params.N * params.H;

    if (block_dim == 0) return;

    // AscendC kernel launch:
    //   FwdRecurrenceKernel<<<blockDim, nullptr, stream>>>(params);

    // Placeholder: actual launch requires CANN SDK headers
    // Same pattern as launch_fwd_prepare
}

}  // namespace flash_kda
