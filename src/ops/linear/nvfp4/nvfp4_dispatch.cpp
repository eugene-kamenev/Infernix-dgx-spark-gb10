#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408, &kNvfp4N16384K2560,
                         &kNvfp4N13952K2560, &kNvfp4N12800K2560, &kNvfp4N2560K6144,
                         &kNvfp4N2048K2560, &kNvfp4N6144K2560, &kNvfp4N512K2560,
                         &kNvfp4N128K2560, &kNvfp4N10240K2560, &kNvfp4N2560K2560,
                         &kNvfp4N1280K2560, &kNvfp4N2560K640, &kNvfp4N13824K2304, &kNvfp4N2304K6912};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy) || column_invariant(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}

bool uses_pair(const Nvfp4LinearShape& shape, LinearPolicy policy, LinearRoutes routes,
               std::int32_t tokens) {
    return (allows_a4_residual(policy) || has_route(routes, LinearRoutes::Nvfp4A4Pair)) &&
           shape.pair != nullptr && tokens >= shape.pair_min_tokens &&
           !(allows_a4(policy) && shape.uses_a4(tokens, tokens));
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    const std::size_t a4 = allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
                               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
                               : 0;
    const std::size_t pair =
        shape.pair != nullptr && max_tokens >= shape.pair_min_tokens
            ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k, kNvfp4A4ResidualTerms, true)
            : 0;
    const std::size_t rht16_a16 =
        static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(k) * sizeof(std::uint16_t);
    return std::max({a4, pair, rht16_a16});
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream, LinearRoutes routes) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (uses_pair(shape, policy, routes, x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("nvfp4 A4 pair linear requires caller workspace");
        auto scope = workspace->scope();
        const auto scratch =
            allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k, kNvfp4A4ResidualTerms, true);
        shape.pair(x, weight, out, scratch, stream);
        return;
    }
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1])) {
        if (weight.layout == QuantLayout::BlockScaleK16M128x4Rht16) {
            if (workspace == nullptr)
                throw std::invalid_argument("nvfp4 RHT-16 A16 linear requires caller workspace");
            auto scope   = workspace->scope();
            Tensor x_rot = workspace->alloc(DType::BF16, {weight.k, x.ne[1]});
            launch_nvfp4_rht16_bf16(x, static_cast<__nv_bfloat16*>(x_rot.data), stream);
            shape.a16(x_rot, weight, out, stream);
            return;
        }
        return shape.a16(x, weight, out, stream);
    }
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace infernix::ops::detail
