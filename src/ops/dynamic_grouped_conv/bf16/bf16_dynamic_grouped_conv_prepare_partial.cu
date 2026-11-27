#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kGroupWidth = 16;

template <int Hidden, class S>
struct GroupedCoefficients : Bf16ScheduleInstance<S, Hidden> {
    static constexpr int kGroups = Hidden / kGroupWidth;
    // Four adjacent coefficient rows belong to one group. Native P remains [side, tap, group].
    __device__ __forceinline__ static int weight_row(int row) {
        return (row % 4) * kGroups + row / 4;
    }
};

template <int Hidden>
struct InputConvolution {
    enum { kSlicedTile = true };
    static constexpr int kGroups = Hidden / kGroupWidth;

    template <class S>
    __device__ __forceinline__ void finish_sliced_tile(DynamicConvFusedOutputs out,
                                                       const float* coefficients, int row0,
                                                       int token0, int, int tokens) const {
        constexpr int groups_per_tile = S::kBlockRows / 4;
        constexpr int pairs_per_tile  = groups_per_tile * (kGroupWidth / 2);
        const int columns             = min(S::kBlockTokens, tokens - token0);
        for (int item = threadIdx.x; item < columns * pairs_per_tile; item += S::kThreads) {
            const int column  = item / pairs_per_tile;
            const int token   = token0 + column;
            const int local   = item - column * pairs_per_tile;
            const int g       = local / (kGroupWidth / 2);
            const int h       = row0 / 4 * kGroupWidth + local * 2;
            const auto offset = static_cast<std::int64_t>(token) * Hidden + h;
            const auto c      = column * S::kBlockRows + 4 * g;
            const float d0    = coefficients[c];
            const float2 base0 =
                __bfloat1622float2(load_vec<__nv_bfloat162>(out.base + h));
            const float2 norm0 =
                __bfloat1622float2(load_vec<__nv_bfloat162>(out.normalized + offset));
            float2 value{(base0.x + d0) * norm0.x, (base0.y + d0) * norm0.y};
            if (token % out.width) {
                const float d1 = coefficients[c + 1];
                const float2 base1 =
                    __bfloat1622float2(load_vec<__nv_bfloat162>(out.base + Hidden + h));
                const float2 norm1 = __bfloat1622float2(
                    load_vec<__nv_bfloat162>(out.normalized + offset - Hidden));
                value.x = fmaf(base1.x + d1, norm1.x, value.x);
                value.y = fmaf(base1.y + d1, norm1.y, value.y);
            }
            store_vec(out.prepared + offset, __floats2bfloat162_rn(value.x, value.y));
        }
        for (int item = threadIdx.x; item < columns * groups_per_tile * 2; item += S::kThreads) {
            const int group  = item % groups_per_tile;
            const int tap    = (item / groups_per_tile) % 2;
            const int column = item / (groups_per_tile * 2);
            const int token  = token0 + column;
            const auto offset =
                (static_cast<std::int64_t>(token) * 2 + tap) * kGroups + row0 / 4 + group;
            out.finish[offset] =
                __float2bfloat16_rn(coefficients[column * S::kBlockRows + group * 4 + 2 + tap]);
        }
    }
};

template <int Hidden, int Columns, int Warps, int WarpK>
void run_sliced(const Tensor& normalized, const Tensor& base, const Weight& projection,
                Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    using S = GroupedCoefficients<
        Hidden, Bf16A16SlicedKMmaSchedule<16, Columns, Warps, WarpK, 2, Cache::cg, Cache::cg, 1, 7>>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(normalized.data),
                            static_cast<const __nv_bfloat16*>(projection.qdata), Hidden / 4, Hidden,
                            normalized.ne[1] * normalized.ne[2]};
    const DynamicConvFusedOutputs out{static_cast<const __nv_bfloat16*>(normalized.data),
                                      static_cast<const __nv_bfloat16*>(base.data),
                                      static_cast<__nv_bfloat16*>(prepared.data),
                                      static_cast<__nv_bfloat16*>(finish.data), normalized.ne[1]};
    launch_bf16_a16_sliced_k_mma<S>(p, out, InputConvolution<Hidden>{}, stream,
                                    pdl::Dependency::Programmatic);
}

template <int Hidden>
void launch_hidden_sliced(const Tensor& normalized, const Tensor& base, const Weight& projection,
                          Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const int columns  = normalized.ne[1] * normalized.ne[2];
    constexpr int Wide = Hidden % 512 == 0 ? 16 : 8;
    if (columns <= 8)
        return run_sliced<Hidden, 8, Wide, 32>(normalized, base, projection, prepared, finish, stream);
    if (columns <= 16)
        return run_sliced<Hidden, 16, Wide, 32>(normalized, base, projection, prepared, finish,
                                                stream);
    if (columns <= 32)
        return run_sliced<Hidden, 32, Wide, 32>(normalized, base, projection, prepared, finish,
                                                stream);
    if (columns <= 64)
        return run_sliced<Hidden, 64, 8, 32>(normalized, base, projection, prepared, finish, stream);
    throw std::logic_error("dynamic grouped conv prepare: sliced route requires columns <= 64");
}

} // namespace

void bf16_dynamic_grouped_conv_prepare_fused_sliced_launch(const Tensor& normalized,
                                                           const Tensor& base,
                                                           const Weight& projection,
                                                           Tensor& prepared, Tensor& finish,
                                                           cudaStream_t stream) {
    switch (normalized.ne[0]) {
    case 5120:
        return launch_hidden_sliced<5120>(normalized, base, projection, prepared, finish, stream);
    case 2560:
        return launch_hidden_sliced<2560>(normalized, base, projection, prepared, finish, stream);
    case 3072:
        return launch_hidden_sliced<3072>(normalized, base, projection, prepared, finish, stream);
    case 2304:
        return launch_hidden_sliced<2304>(normalized, base, projection, prepared, finish, stream);
    }
    throw std::logic_error("dynamic grouped conv prepare: unregistered hidden width");
}

} // namespace infernix::ops::detail
