#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "ops/linear/bf16/bf16_a16_tma_register_b.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

constexpr int kGroupWidth = 16;

// A 4D view of the existing [plane, group, K] weights loads all four coefficients
// of a contiguous group interval into one tile, without materializing a new layout.
struct NativeCoefficientAccess {
    template <class S>
    static CUtensorMap map(const Bf16A16Operands& p) {
        CUtensorMap result{};
        std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(p.k / 64),
                                   static_cast<std::uint64_t>(p.rows / 4), 4};
        std::uint64_t strides[]{128, static_cast<std::uint64_t>(p.k) * 2,
                                static_cast<std::uint64_t>(p.rows / 4) * p.k * 2};
        std::uint32_t box[]{64, static_cast<std::uint32_t>(S::kBlockK / 64),
                            static_cast<std::uint32_t>(S::kBlockRows / 4), 4};
        const std::uint32_t steps[]{1, 1, 1, 1};
        const auto status = cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(p.weight),
            dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(status, &name);
            throw std::runtime_error(std::string("dynamic grouped conv prepare TensorMap: ") +
                                     (name ? name : "CUDA error"));
        }
        return result;
    }

    template <class S>
    __device__ __forceinline__ static void load(void* destination, const CUtensorMap* map,
                                                int k_sector, int row, std::uint64_t* barrier) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                     :
                     : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector),
                       "r"(row / 4), "r"(0), "r"(smem_addr(barrier))
                     : "memory");
    }
};

template <int Hidden>
struct CoefficientConvolution {
    enum { kTmaCollective = true };
    static constexpr int kGroups = Hidden / kGroupWidth;

    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * sizeof(float);

    template <class S, bool FullTokens, class Output>
    __device__ __forceinline__ void finish_tile(const Output& out, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int,
                                                int tokens) const {
        constexpr int rows   = S::kBlockRows;
        constexpr int groups = rows / 4;
        constexpr int pairs  = groups * (kGroupWidth / 2);
        static_assert(rows >= 32 && (rows & (rows - 1)) == 0);
        auto* tile    = reinterpret_cast<float*>(scratch);
        const auto at = [&](int row, int token) {
            return tile + token * rows + (row ^ (((token / 2) & (rows / 8 - 1)) * 8));
        };
        const auto write_tile = [&](int tid) {
            const int warp     = tid / 32;
            const int lane     = tid % 32;
            const int warp_row = warp / S::kWarpsTokens;
            const int warp_tok = warp % S::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row = warp_row * S::kWarpRows + mi * 16 + lane / 4;
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int token         = warp_tok * S::kWarpTokens + ni * 8 + 2 * (lane % 4);
                    *at(row, token)         = accum[mi][ni][0];
                    *at(row, token + 1)     = accum[mi][ni][1];
                    *at(row + 8, token)     = accum[mi][ni][2];
                    *at(row + 8, token + 1) = accum[mi][ni][3];
                }
            }
        };
        if constexpr (S::kProducerThreads == 0) {
            write_tile(static_cast<int>(threadIdx.x));
        } else if (threadIdx.x >= S::kProducerThreads) {
            write_tile(static_cast<int>(threadIdx.x) - S::kProducerThreads);
        }
        __syncthreads();
        const int columns =
            FullTokens ? S::kBlockTokens : min(S::kBlockTokens, tokens - token_begin);
        for (int item = threadIdx.x; item < columns * pairs; item += S::kThreads) {
            const int column  = item / pairs;
            const int local   = item - column * pairs;
            const int group   = local / (kGroupWidth / 2);
            const int hidden  = row_begin / 4 * kGroupWidth + local * 2;
            const int token   = token_begin + column;
            const auto offset = static_cast<std::int64_t>(token) * Hidden + hidden;
            const float d0    = *at(group, column);
            const float2 base0 =
                __bfloat1622float2(load_vec<__nv_bfloat162>(out.base + hidden));
            const float2 norm0 =
                __bfloat1622float2(load_vec<__nv_bfloat162>(out.normalized + offset));
            float2 value{(base0.x + d0) * norm0.x, (base0.y + d0) * norm0.y};
            if (token % out.width) {
                const float d1 = *at(groups + group, column);
                const float2 base1 =
                    __bfloat1622float2(load_vec<__nv_bfloat162>(out.base + Hidden + hidden));
                const float2 norm1 = __bfloat1622float2(
                    load_vec<__nv_bfloat162>(out.normalized + offset - Hidden));
                value.x = fmaf(base1.x + d1, norm1.x, value.x);
                value.y = fmaf(base1.y + d1, norm1.y, value.y);
            }
            store_vec(out.prepared + offset, __floats2bfloat162_rn(value.x, value.y));
        }
        for (int item = threadIdx.x; item < columns * groups * 2; item += S::kThreads) {
            const int column  = item / (groups * 2);
            const int tap     = (item / groups) % 2;
            const int group   = item % groups;
            const auto offset = (static_cast<std::int64_t>(token_begin + column) * 2 + tap) *
                                    kGroups +
                                row_begin / 4 + group;
            out.finish[offset] = __float2bfloat16_rn(*at((2 + tap) * groups + group, column));
        }
    }
};

template <int Hidden, int Rows, int Tokens, int K, int WarpRows, int WarpTokens, int Stages,
          bool RegisterB = false, Bf16MmaRaster Raster = Bf16MmaRaster::TokenFast>
void launch_tma_fused(const Tensor& normalized, const Tensor& base, const Weight& weight,
                      Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    using Base = std::conditional_t<
        RegisterB, Bf16A16TmaRegisterBSchedule<Rows, Tokens, K, WarpRows, WarpTokens, 2, Raster>,
        Bf16A16TmaMmaSchedule<Rows, Tokens, K, WarpRows, WarpTokens, Stages>>;
    using S = Bf16ScheduleInstance<Base, Hidden>;
    const Bf16A16Operands operands{static_cast<const __nv_bfloat16*>(normalized.data),
                                   static_cast<const __nv_bfloat16*>(weight.qdata), Hidden / 4,
                                   Hidden, normalized.ne[1] * normalized.ne[2]};
    const DynamicConvFusedOutputs output{static_cast<const __nv_bfloat16*>(normalized.data),
                                         static_cast<const __nv_bfloat16*>(base.data),
                                         static_cast<__nv_bfloat16*>(prepared.data),
                                         static_cast<__nv_bfloat16*>(finish.data),
                                         normalized.ne[1]};
    if constexpr (RegisterB) {
        launch_bf16_a16_tma_register_b<S, DynamicConvFusedOutputs, CoefficientConvolution<Hidden>,
                                       NativeCoefficientAccess>(operands, output, {}, stream,
                                                                pdl::Dependency::Programmatic);
    } else {
        launch_bf16_a16_tma_mma<S, 1, DynamicConvFusedOutputs, CoefficientConvolution<Hidden>,
                                NativeCoefficientAccess>(operands, output, {}, stream,
                                                         pdl::Dependency::Programmatic);
    }
}

template <int Hidden>
void launch_hidden_tma(const Tensor& normalized, const Tensor& base, const Weight& projection,
                       Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const int columns = normalized.ne[1] * normalized.ne[2];
    if (columns <= 128) {
        return launch_tma_fused<Hidden, 32, 32, 128, 16, 16, 2, true>(normalized, base, projection,
                                                                      prepared, finish, stream);
    }
    if (columns <= 256) {
        return launch_tma_fused<Hidden, 32, 64, 64, 16, 32, 3>(normalized, base, projection,
                                                               prepared, finish, stream);
    }
    if (columns < 512) {
        return launch_tma_fused<Hidden, 64, 64, 64, 32, 32, 2, true>(normalized, base, projection,
                                                                     prepared, finish, stream);
    }
    if (columns >= 1024) {
        return launch_tma_fused<Hidden, 64, 128, 64, 32, 32, 2, true, Bf16MmaRaster::RowFast>(
            normalized, base, projection, prepared, finish, stream);
    }
    launch_tma_fused<Hidden, 64, 128, 64, 32, 32, 2, true>(normalized, base, projection, prepared,
                                                           finish, stream);
}

} // namespace

void bf16_dynamic_grouped_conv_prepare_fused_tma_launch(const Tensor& normalized,
                                                        const Tensor& base,
                                                        const Weight& projection, Tensor& prepared,
                                                        Tensor& finish, cudaStream_t stream) {
    switch (normalized.ne[0]) {
    case 5120:
        return launch_hidden_tma<5120>(normalized, base, projection, prepared, finish, stream);
    case 2560:
        return launch_hidden_tma<2560>(normalized, base, projection, prepared, finish, stream);
    case 3072:
        return launch_hidden_tma<3072>(normalized, base, projection, prepared, finish, stream);
    case 2304:
        return launch_hidden_tma<2304>(normalized, base, projection, prepared, finish, stream);
    }
    throw std::logic_error("dynamic grouped conv prepare: unregistered hidden width");
}

} // namespace infernix::ops::detail
