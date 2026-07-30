/**
 * FlashKDA Ascend — Kernel 1 Launch (Prepare)
 *
 * Host-side kernel launch for FwdPrepareKernel.
 * Grid: (total_tiles * H) — one AIC+AIV core per (tile, head)
 */

#include "flash_kda/fwd.h"
#include "flash_kda/fwd_kernel1.hpp"

namespace flash_kda {

void launch_fwd_prepare(const FwdParams& params, aclrtStream stream) {
    // Grid dimension: total_tiles * H blocks
    // Each block = 1 AIC core + 1 AIV core (AscendC auto-schedules)
    int block_dim = params.total_tiles * params.H;

    if (block_dim == 0) return;

    // AscendC kernel launch:
    //   KernelName<<<blockDim, nullptr, stream>>>(params)
    //
    // The FwdPrepareKernel class has operator()<AIC> and operator()<AIV>
    // specializations. AscendC runtime automatically invokes both
    // on the same physical core (AIC + AIV sub-cores).
    //
    // On Ascend, the kernel is launched as:
    //   FwdPrepareKernel<<<block_dim, nullptr, stream>>>(params)

    // Note: AscendC kernel launch syntax differs from CUDA.
    // The actual launch uses ACL runtime:
    //   aclrtLaunchKernel<FwdPrepareKernel>(block_dim, 1, stream, params);

    // For CANN SDK compatibility, use the macro-based launch:
    // LAUNCH_KERNEL(FwdPrepareKernel, block_dim, stream, params);

    // Placeholder: actual launch requires CANN SDK headers
    // The pattern from catlass FAI example:
    //   FAInferFp16<<<blockDim, nullptr, stream>>>(hardwareSyncAddr, ...);
    //
    // For our kernel:
    //   FwdPrepareKernel<<<blockDim, nullptr, stream>>>(params);
}

}  // namespace flash_kda
