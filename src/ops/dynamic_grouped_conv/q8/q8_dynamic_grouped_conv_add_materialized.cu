#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// One 16-byte (8-channel) slice of the projected column per finish_kernel thread.
constexpr int kFinishChannels = 8;
constexpr int kFinishThreads  = 256;

using Launch = Q8Launch;

template <int Rows, int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // K warps take 64-wide slices: 4 where C does not divide into 512s (Inkling 2 Mini's 6912).
    constexpr int Warps = InputRows % 512 != 0     ? 4
                          : InputRows == 4096      ? (TileColumns <= 40 ? 8 : 4)
                                                   : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    constexpr int MinBlocks = (Warps == 8 || TileColumns > 56) ? 2 : 3;
    using Geometry          = Q8LinearGeometry<Rows, InputRows>;
    using Schedule          = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, MinBlocks,
                                                      Q8ScaleAccess::Shared, Activation>;
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), Rows};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream, {}, pdl::Dependency::Programmatic);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int Rows, int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &tiled_projection<Rows, C, 8 * (1 + static_cast<int>(I))>...};
}

// The registered (H, C) pairs: Qwen3.8-27B's drafter (5120; 4096 and 17408),
// Qwen3.8-Flash-Next's (2560; 4096 and 7680) and Qwen3.5-122B-A10B's (3072; 4096 and 9216).
constexpr auto attention_5120 = make_launchers<5120, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_5120       = make_launchers<5120, 17408>(std::make_index_sequence<11>{});
constexpr auto attention_2560 = make_launchers<2560, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_2560       = make_launchers<2560, 7680>(std::make_index_sequence<11>{});
constexpr auto attention_3072 = make_launchers<3072, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_3072       = make_launchers<3072, 9216>(std::make_index_sequence<11>{});
constexpr auto attention_2304 = make_launchers<2304, 4096>(std::make_index_sequence<11>{});
constexpr auto mlp_2304       = make_launchers<2304, 6912>(std::make_index_sequence<11>{});

// Each thread folds one 16-byte slice of the projected column into the residual. Rows is a
// multiple of 16, so an eight-channel slice that starts on a multiple of 8 never crosses a
// 16-channel group boundary and both delta taps stay scalar.
template <int Rows>
__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    constexpr int kGroups = Rows / 16;
    pdl::enter();
    const int row = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x)) * kFinishChannels;
    const int col = blockIdx.y;
    if (row >= Rows) return;
    const int index = col * Rows + row, di = col * 2 * kGroups + row / 16;
    const bool has_previous = col % width != 0;
    const float current_tap = __bfloat162float(delta[di]);
    const float previous_tap = has_previous ? __bfloat162float(delta[di + kGroups]) : 0.0f;

    const uint4 current_bits   = *reinterpret_cast<const uint4*>(projected + index);
    const uint4 base_bits      = *reinterpret_cast<const uint4*>(base + 2 * Rows + row);
    const uint4 residual_bits  = *reinterpret_cast<const uint4*>(residual + index);
    const uint4 previous_bits  = has_previous
                                     ? *reinterpret_cast<const uint4*>(projected + index - Rows)
                                     : make_uint4(0U, 0U, 0U, 0U);
    const uint4 prior_base_bits = has_previous
                                      ? *reinterpret_cast<const uint4*>(base + 3 * Rows + row)
                                      : make_uint4(0U, 0U, 0U, 0U);
    const auto* current  = reinterpret_cast<const __nv_bfloat162*>(&current_bits);
    const auto* previous = reinterpret_cast<const __nv_bfloat162*>(&previous_bits);
    const auto* tap      = reinterpret_cast<const __nv_bfloat162*>(&base_bits);
    const auto* prior    = reinterpret_cast<const __nv_bfloat162*>(&prior_base_bits);
    const auto* carried  = reinterpret_cast<const __nv_bfloat162*>(&residual_bits);

    uint4 result;
    auto* folded = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int pair = 0; pair < kFinishChannels / 2; ++pair) {
        float2 value = __bfloat1622float2(carried[pair]);
        const float2 current_coefficient = __bfloat1622float2(tap[pair]);
        const float2 current_projection  = __bfloat1622float2(current[pair]);
        value.x = fmaf(current_coefficient.x + current_tap, current_projection.x, value.x);
        value.y = fmaf(current_coefficient.y + current_tap, current_projection.y, value.y);
        if (has_previous) {
            const float2 previous_coefficient = __bfloat1622float2(prior[pair]);
            const float2 previous_projection  = __bfloat1622float2(previous[pair]);
            value.x = fmaf(previous_coefficient.x + previous_tap, previous_projection.x, value.x);
            value.y = fmaf(previous_coefficient.y + previous_tap, previous_projection.y, value.y);
        }
        folded[pair] = __floats2bfloat162_rn(value.x, value.y);
    }
    *reinterpret_cast<uint4*>(residual + index) = result;
}

const std::array<Launch, 11>& tiled_launchers(int rows, int input_rows) {
    if (rows == 5120 && input_rows == 4096) return attention_5120;
    if (rows == 5120 && input_rows == 17408) return mlp_5120;
    if (rows == 2560 && input_rows == 4096) return attention_2560;
    if (rows == 2560 && input_rows == 7680) return mlp_2560;
    if (rows == 3072 && input_rows == 4096) return attention_3072;
    if (rows == 3072 && input_rows == 9216) return mlp_3072;
    if (rows == 2304 && input_rows == 4096) return attention_2304;
    if (rows == 2304 && input_rows == 6912) return mlp_2304;
    throw std::logic_error("linear dynamic grouped conv add: unregistered (H, C)");
}

template <int Rows>
void finish(const Tensor& projected, const Tensor& base, const Tensor& delta, Tensor& residual,
            int width, int tokens, cudaStream_t stream) {
    const dim3 grid((Rows / kFinishChannels + kFinishThreads - 1) / kFinishThreads, tokens);
    CUDA_CHECK(pdl::launch_consumer({grid, dim3(kFinishThreads), 0, stream}, finish_kernel<Rows>,
                                    static_cast<const __nv_bfloat16*>(projected.data),
                                    static_cast<const __nv_bfloat16*>(base.data),
                                    static_cast<const __nv_bfloat16*>(delta.data),
                                    static_cast<__nv_bfloat16*>(residual.data), width));
}

void finish_rows(const Tensor& projected, const Tensor& base, const Tensor& delta, Tensor& residual,
                 int width, int tokens, cudaStream_t stream) {
    const int rows = residual.ne[0];
    if (rows == 5120) {
        finish<5120>(projected, base, delta, residual, width, tokens, stream);
    } else if (rows == 3072) {
        finish<3072>(projected, base, delta, residual, width, tokens, stream);
    } else if (rows == 2304) {
        finish<2304>(projected, base, delta, residual, width, tokens, stream);
    } else {
        finish<2560>(projected, base, delta, residual, width, tokens, stream);
    }
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int rows    = residual.ne[0];
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({rows, tokens});
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma:
        tiled_launchers(rows, x.ne[0])[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    case Q8DynamicConvAddSchedule::MmaK128:
        launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        break;
    case Q8DynamicConvAddSchedule::MmaT128:
        launch_q8_a16_mma_r64_t128(flat, weight, result, stream);
        break;
    }
    finish_rows(projected, base, delta, residual, x.ne[1], tokens, stream);
}
} // namespace

void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base,
                                            const Tensor& delta, Tensor& residual, int width,
                                            int tokens, cudaStream_t stream) {
    finish_rows(projected, base, delta, residual, width, tokens, stream);
}

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace infernix::ops::detail
