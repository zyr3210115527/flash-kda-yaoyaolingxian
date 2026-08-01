/**
 * FlashKDA Ascend — Host-side pybind11 binding + ACL runtime
 *
 * Mirrors the CUDA version's flash_kda.cpp:
 *   - Input validation (shapes, dtypes, contiguity)
 *   - Workspace size computation
 *   - ACL stream acquisition
 *   - Kernel launch dispatch
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>
#include <cstdlib>

#include "flash_kda/fwd.h"
#include "flash_kda/layout.hpp"

namespace py = pybind11;
using namespace flash_kda;

namespace {

void check_tensor(const at::Tensor& t, const char* name, torch::ScalarType expected_dtype,
                  bool require_cuda = true) {
    if (require_cuda) {
        TORCH_CHECK(t.is_privateuseone(), name, " must be on NPU");
    }
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == expected_dtype, name, " has wrong dtype");
}

}  // namespace

void fwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor g,
    at::Tensor beta,
    double scale,
    at::Tensor out,
    at::Tensor workspace,
    at::Tensor A_log,
    at::Tensor dt_bias,
    double lower_bound,
    std::optional<at::Tensor> initial_state,
    std::optional<at::Tensor> final_state,
    std::optional<at::Tensor> cu_seqlens
) {
    // Validate dtypes
    check_tensor(q, "q", torch::kBFloat16);
    check_tensor(k, "k", torch::kBFloat16);
    check_tensor(v, "v", torch::kBFloat16);
    check_tensor(g, "g", torch::kBFloat16);
    check_tensor(beta, "beta", torch::kBFloat16);
    check_tensor(out, "out", torch::kBFloat16);
    check_tensor(workspace, "workspace", torch::kUInt8);
    check_tensor(A_log, "A_log", torch::kFloat32);
    check_tensor(dt_bias, "dt_bias", torch::kFloat32);

    // Validate state tensors
    bool has_state_in = initial_state.has_value();
    bool has_state_out = final_state.has_value();
    bool state_fp32 = false;

    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.is_privateuseone() && is.is_contiguous(), "initial_state must be contiguous NPU tensor");
        TORCH_CHECK(is.scalar_type() == torch::kBFloat16 || is.scalar_type() == torch::kFloat32,
                     "initial_state must be bfloat16 or float32");
        if (is.scalar_type() == torch::kFloat32) state_fp32 = true;
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.is_privateuseone() && fs.is_contiguous(), "final_state must be contiguous NPU tensor");
        TORCH_CHECK(fs.scalar_type() == torch::kBFloat16 || fs.scalar_type() == torch::kFloat32,
                     "final_state must be bfloat16 or float32");
        if (fs.scalar_type() == torch::kFloat32) state_fp32 = true;
    }
    if (has_state_in && has_state_out) {
        TORCH_CHECK(initial_state->scalar_type() == final_state->scalar_type(),
                     "initial_state and final_state must have the same dtype");
    }

    // Validate shapes
    TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
    TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
    TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

    int64_t B = q.size(0);
    int64_t T_seq = q.size(1);
    int64_t H = q.size(2);
    int64_t D_dim = q.size(3);
    int64_t T_total = B * T_seq;

    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(g.sizes() == q.sizes(), "g must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
                "beta must be [B, T, H] matching q");

    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D_dim,
                "dt_bias must be [H, D]");

    TORCH_CHECK(D_dim == 128, "currently only supports D == 128");

    // Determine varlen
    bool is_varlen = cu_seqlens.has_value();
    int64_t N_val;
    int64_t const* cu_seqlens_dev = nullptr;

    if (is_varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        auto& cu_seqlens_t = cu_seqlens.value();
        TORCH_CHECK(cu_seqlens_t.is_privateuseone(), "cu_seqlens must be on NPU");
        TORCH_CHECK(cu_seqlens_t.scalar_type() == torch::kLong, "cu_seqlens must be int64");
        TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
        N_val = cu_seqlens_t.numel() - 1;
        TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
        cu_seqlens_dev = cu_seqlens_t.data_ptr<int64_t>();
    } else {
        N_val = B;
    }

    // Validate state shapes: [N, H, D, D]
    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
        TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D_dim && is.size(3) == D_dim,
                     "initial_state must be [N, H, D, D]");
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
        TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D_dim && fs.size(3) == D_dim,
                     "final_state must be [N, H, D, D]");
    }

    // Compute total_tiles
    int total_tiles;
    if (is_varlen) {
        total_tiles = int((T_total + CHUNK - 1) / CHUNK + N_val);
    } else {
        total_tiles = int(N_val * ((T_seq + CHUNK - 1) / CHUNK));
    }

    // Flatten [B, T, H, D] -> [T_total, H, D] (contiguous, same data pointer)
    // This matches the CUDA version's layout and the kernel's GM offset calculation:
    //   element (h, t, d) at offset h * T_total * D + t * D + d
    auto q_3d = q.reshape({T_total, H, D_dim});
    auto k_3d = k.reshape({T_total, H, D_dim});
    auto v_3d = v.reshape({T_total, H, D_dim});
    auto g_3d = g.reshape({T_total, H, D_dim});
    auto out_3d = out.reshape({T_total, H, D_dim});

    // beta stays in its native [T_total, H] layout; the kernel reads a chunk's
    // 16 values with a stride of H. Transposing here would materialize a
    // device-side copy for no benefit.
    auto beta_2d = beta.reshape({T_total, H});

    // Build params
    FwdParams params{};
    params.q = reinterpret_cast<GM_ADDR>(q_3d.data_ptr());
    params.k = reinterpret_cast<GM_ADDR>(k_3d.data_ptr());
    params.v = reinterpret_cast<GM_ADDR>(v_3d.data_ptr());
    params.g = reinterpret_cast<GM_ADDR>(g_3d.data_ptr());
    params.beta = reinterpret_cast<GM_ADDR>(beta_2d.data_ptr());
    params.A_log = reinterpret_cast<GM_ADDR>(A_log.data_ptr());
    params.dt_bias = reinterpret_cast<GM_ADDR>(dt_bias.data_ptr());
    params.out = reinterpret_cast<GM_ADDR>(out_3d.data_ptr());
    params.workspace = reinterpret_cast<GM_ADDR>(workspace.data_ptr());
    params.initial_state = has_state_in ? reinterpret_cast<GM_ADDR>(initial_state->data_ptr()) : nullptr;
    params.final_state = has_state_out ? reinterpret_cast<GM_ADDR>(final_state->data_ptr()) : nullptr;
    params.cu_seqlens =
        reinterpret_cast<GM_ADDR>(const_cast<int64_t *>(cu_seqlens_dev));
    params.scale = static_cast<float>(scale);
    // Raw lower_bound, no log2(e). The CUDA kernel folds log2(e) in because it
    // decays with ex2 (2^x); this port uses the natural Exp, so pre-multiplying
    // here would inflate every decay exponent by 1.4427x.
    params.gate_scale = static_cast<float>(lower_bound);
    params.T_total = int(T_total);
    params.H = int(H);
    params.N = int(N_val);
    params.total_tiles = total_tiles;
    params.has_state_in = has_state_in ? 1 : 0;
    params.has_state_out = has_state_out ? 1 : 0;
    // Keep the Neumann chain in L1 rather than round-tripping GM. Read once;
    // the chain is the largest single item in the pipeline (8.7 ms of 30 at
    // T=4096 H=64) and almost all of that is six GM round trips.
    static const int kL1Neumann = [] {
        const char *e = std::getenv("FLASH_KDA_L1_NEUMANN");
        // Default on: measured 23.7 ms against 30.3 at T=4096 H=64, with
        // byte-identical error values. FLASH_KDA_L1_NEUMANN=0 selects the GM
        // path, which is the reference it was validated against.
        return (e == nullptr || e[0] != '0') ? 1 : 0;
    }();
    params.l1_neumann = kL1Neumann;
    params.state_fp32 = state_fp32 ? 1 : 0;
    params.is_varlen = is_varlen ? 1 : 0;
    params.chunk_idx = 0;
    params.state_ws_offset = flash_kda::get_state_ws_offset(T_total, H, N_val);

    // Get ACL stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    // Launch kernels
    if (std::getenv("FLASH_KDA_SYNC_SMALL") != nullptr) {
        launch_sync_small(total_tiles * int(H), stream);
        return;
    }
    if (std::getenv("FLASH_KDA_SYNC_ONLY") != nullptr) {
        launch_sync_only(params, stream);
        return;
    }
    launch_fwd_prepare(params, stream);
    // Kernel2 still synchronizes its AIC and AIV halves with CrossCore flags,
    // which deadlock when the kernel is launched from a Python extension (see
    // docs/debugging-notes.md). Until it gets the same split kernel1 received,
    // this lets kernel1 be exercised on its own.
    if (std::getenv("FLASH_KDA_SKIP_K2") == nullptr) {
        launch_fwd_recurrence(params, stream);
    }
}

int64_t noop(at::Tensor flag) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    flash_kda::launch_noop(reinterpret_cast<GM_ADDR>(flag.data_ptr()), stream);
    return 0;
}

int64_t aiv_only(at::Tensor src, at::Tensor dst) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    flash_kda::launch_aiv_only(reinterpret_cast<GM_ADDR>(src.data_ptr()),
                               reinterpret_cast<GM_ADDR>(dst.data_ptr()), stream);
    return 0;
}

int64_t aic_only(at::Tensor flag) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    flash_kda::launch_aic_only(reinterpret_cast<GM_ADDR>(flag.data_ptr()), stream);
    return 0;
}

int64_t sync_onearg(int64_t n, int64_t h) {
    FwdParams params{};
    params.N = int(n);
    params.H = int(h);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    flash_kda::launch_sync_onearg(params, stream);
    return 0;
}

PYBIND11_MODULE(_C, m) {
    m.def("sync_onearg", &sync_onearg, "single-argument cross-core handshake probe",
          py::arg("n"), py::arg("h"));
    m.def("aic_only", &aic_only, "AIC-only probe", py::arg("flag"));
    m.def("aiv_only", &aiv_only, "AIV-only vector kernel probe",
          py::arg("src"), py::arg("dst"));
    m.def("noop", &noop, "no-op kernel launch probe", py::arg("flag"));
    m.def("fwd", &fwd, "FlashKDA Forward (Ascend)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("scale"), py::arg("out"),
        py::arg("workspace"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("initial_state") = py::none(), py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
    m.def("get_workspace_size",
        static_cast<int64_t (*)(int64_t, int64_t, int64_t)>(
            &flash_kda::get_workspace_size),
        "Get workspace size in bytes",
        py::arg("T_total"), py::arg("H"),
        py::arg("N") = 1);
}
