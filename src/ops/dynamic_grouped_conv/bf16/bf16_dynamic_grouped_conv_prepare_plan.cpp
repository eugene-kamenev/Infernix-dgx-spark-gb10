#include "core/weight.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "infernix/ops/rmsnorm.h"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::size_t workspace_bytes(std::int32_t hidden, std::int32_t tokens) noexcept {
    return static_cast<std::size_t>(hidden) * static_cast<std::size_t>(tokens) *
           sizeof(std::uint16_t);
}

} // namespace

std::size_t bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(std::int32_t hidden,
                                                                       std::int32_t min_width,
                                                                       std::int32_t max_width,
                                                                       std::int32_t min_batch,
                                                                       std::int32_t max_batch) {
    if (hidden != 5120 && hidden != 2560 && hidden != 3072 && hidden != 2304) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare workspace: H must be 5120, 2560, 3072 or 2304");
    }
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch) {
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid W/B interval");
    }
    return workspace_bytes(hidden, max_width * max_batch);
}

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base, const Weight& weight,
                                                Tensor& prepared, Tensor& finish,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope              = workspace.scope();
    const std::int32_t rows = residual.ne[0];
    const std::int32_t cols = residual.ne[1] * residual.ne[2];
    const auto span         = workspace.alloc_bytes(workspace_bytes(rows, cols));
    Tensor normalized(span.data, DType::BF16, {rows, residual.ne[1], residual.ne[2]});
    rmsnorm(residual, norm, eps, false, normalized, stream);
    if (cols <= 64) {
        bf16_dynamic_grouped_conv_prepare_fused_sliced_launch(normalized, base, weight, prepared,
                                                              finish, stream);
    } else {
        bf16_dynamic_grouped_conv_prepare_fused_tma_launch(normalized, base, weight, prepared,
                                                           finish, stream);
    }
}

} // namespace infernix::ops::detail
