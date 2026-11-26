#pragma once

// Single-term block-scaled MXFP8 (W8A8, block-32 E4M3 + UE8M0) prefill GEMM on SM12x.
// Quantizes BF16 activations into a single E4M3 plane + packed UE8M0 scale words (one per 32 K
// values), then executes mma.sync.aligned.m16n8k32...kind::mxf8f6f4.scale_vec::1X.ue8m0 across
// Grouped-8 TMA stages, halving activation SMEM/TMA/MMA traffic relative to Fp8A8PairBlock32.

#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {

struct Mxfp8A8Operands {
    const std::uint8_t* x;                   // single E4M3 codes [T][K]
    const std::uint32_t* x_block_scales;     // packed UE8M0 words [K / 128][stride]
    const std::uint8_t* codes;               // weight E4M3 codes [N][K]
    const std::uint8_t* weight_block_scales; // weight UE8M0 plane [N/16][K/32][16]
    int rows, k, tokens;
};

template <class Base>
struct Mxfp8A8TmaSchedule : Base {
    static_assert(Base::kSplitWaveCtas == 0, "mxfp8-a8 has no split-K plan");
    static_assert(Base::kBlockK == kFp8A8PairBlock32K,
                  "a stage's scale word holds its K tile's four 32-value blocks");
    static constexpr int kScaleStageBytes = Base::kBlockTokens * 4;
    static constexpr int kCodeBytes =
        Base::kStages * (Base::kBlockTokens + Base::kBlockRows) * Base::kBlockK;
    static constexpr int kPipelineBytes      = kCodeBytes + Base::kStages * kScaleStageBytes;
    static constexpr int kOutputStagingBytes = Base::kBlockTokens * (Base::kBlockRows + 8) * 2;
    static constexpr int kStorageBytes =
        kPipelineBytes > kOutputStagingBytes ? kPipelineBytes : kOutputStagingBytes;
    static constexpr int kSharedBytes = kStorageBytes + Base::kBarrierBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

using Mxfp8A8T64R64G8S3 =
    Mxfp8A8TmaSchedule<Fp8A8TmaMmaSchedule<64, 64, 128, 2, 2, 3, 1, Fp8MmaRaster::Grouped, 8>>;
using Mxfp8A8T64R128G8S3 =
    Mxfp8A8TmaSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>>;
using Mxfp8A8T128R64G8S3 =
    Mxfp8A8TmaSchedule<Fp8A8TmaMmaSchedule<128, 64, 128, 2, 2, 3, 1, Fp8MmaRaster::Grouped, 8>>;
using Mxfp8A8T128R128G8S3 =
    Mxfp8A8TmaSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1, Fp8MmaRaster::Grouped, 8>>;
using Mxfp8A8T64R256G8S2 =
    Mxfp8A8TmaSchedule<Fp8A8TmaMmaSchedule<64, 256, 128, 2, 8, 2, 1, Fp8MmaRaster::Grouped, 8>>;

struct alignas(128) Mxfp8A8TmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
    CUtensorMap x_scales;
};

#ifdef _WIN32
inline TmaDescriptorStaging<Mxfp8A8TmaDescriptors>& mxfp8_a8_descriptor_staging() {
    static TmaDescriptorStaging<Mxfp8A8TmaDescriptors> staging;
    return staging;
}
#endif

template <class Schedule, int KStep>
__device__ __forceinline__ void mxfp8_mma_a8_k_step(
    const std::uint8_t* activation, const std::uint8_t* weight_stage,
    const unsigned (&scale_a)[Schedule::kMmaTokens], const unsigned (&scale_b)[Schedule::kMmaRows],
    float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp, int lane) {
    constexpr int BK        = Schedule::kBlockK;
    constexpr int Tokens    = Schedule::kMmaTokens;
    constexpr int Rows      = Schedule::kMmaRows;
    const int warp_token    = warp / Schedule::kWarpsRows;
    const int warp_row      = warp % Schedule::kWarpsRows;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    unsigned b[Rows][2];
    unsigned a[Tokens][4];
#pragma unroll
    for (int mma_row = 0; mma_row < Rows; ++mma_row) {
        const int row           = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
        const int physical_byte = fp8_mma_shared_byte<Schedule>(row, KStep * 32 + b_column_byte);
        ldmatrix_x2(b[mma_row][0], b[mma_row][1], smem_addr(weight_stage + row * BK + physical_byte));
    }
#pragma unroll
    for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
        const int row           = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
        const int physical_byte = fp8_mma_shared_byte<Schedule>(row, KStep * 32 + a_column_byte);
        ldmatrix_x4(a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                    smem_addr(activation + row * BK + physical_byte));
    }
#pragma unroll
    for (int mma_token = 0; mma_token < Tokens; ++mma_token) {
#pragma unroll
        for (int mma_row = 0; mma_row < Rows; ++mma_row) {
            mma_fp8_e4m3_block_scaled<KStep, KStep>(
                accumulators[mma_token][mma_row][0], accumulators[mma_token][mma_row][1],
                accumulators[mma_token][mma_row][2], accumulators[mma_token][mma_row][3],
                a[mma_token][0], a[mma_token][1], a[mma_token][2], a[mma_token][3],
                b[mma_row][0], b[mma_row][1], scale_a[mma_token], scale_b[mma_row]);
        }
    }
}

template <class Schedule, int... KSteps>
__device__ __forceinline__ void mxfp8_mma_a8_compute_stage(
    const std::uint8_t* activation, const std::uint8_t* weight_stage,
    const unsigned (&scale_a)[Schedule::kMmaTokens], const unsigned (&scale_b)[Schedule::kMmaRows],
    float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp, int lane,
    std::integer_sequence<int, KSteps...>) {
    static_assert(sizeof...(KSteps) == 4, "a 128-value K tile is four k32 steps, one scale byte each");
    (mxfp8_mma_a8_k_step<Schedule, KSteps>(activation, weight_stage, scale_a, scale_b, accumulators,
                                           warp, lane),
     ...);
}

template <class Schedule, bool FullTokens, class Output, bool InPlace = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
mxfp8_a8_tma_mma_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<Mxfp8A8TmaDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ Mxfp8A8TmaDescriptors descriptors,
#endif
    Mxfp8A8Operands operands, Output output, int token_offset, int count) {
#ifdef _WIN32
    const Mxfp8A8TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    constexpr int plane_bytes = S * BT * BK;
    const int k               = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tiles_k         = k / BK;
    const int row_tiles = operands.rows / BR, token_tiles = div_up(count, BT);
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(static_cast<std::int32_t>(blockIdx.x), row_tiles,
                                       token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(128) unsigned char mxfp8_a8_shared[];
    auto* activation = mxfp8_a8_shared;
    auto* weight     = activation + plane_bytes;
    auto* x_scales =
        reinterpret_cast<std::uint32_t*>(mxfp8_a8_shared + Schedule::kCodeBytes);
    auto* full = reinterpret_cast<std::uint64_t*>(
        mxfp8_a8_shared + fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
            acquire_staged_tensor_map<InPlace>(&descriptors.x_scales);
#endif
#pragma unroll 1
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage,
                                              (BT + BR) * BK + Schedule::kScaleStageBytes);
                fp8_tma_load(activation + stage * BT * BK, &descriptors.activation, kt * BK,
                             token_begin, full + stage);
                fp8_tma_load(weight + stage * BR * BK, &descriptors.weight, kt * BK, row_begin,
                             full + stage);
                fp8_tma_load(x_scales + stage * BT, &descriptors.x_scales, token_begin, kt,
                             full + stage);
            }
        }
        return;
    }

    const int tid         = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp        = tid >> 5, lane = tid & 31;
    const int scale_token = warp / Schedule::kWarpsRows * Schedule::kWarpTokens +
                            (((lane & 1) << 3) | (lane >> 2));
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    unsigned weight_scales[Schedule::kMmaRows]                      = {};
    unsigned next_weight_scales[Schedule::kMmaRows]                 = {};
    fp8_pair_block32_weight_scales<Schedule>(operands.weight_block_scales, k, row_begin, 0, warp,
                                             lane, weight_scales);
#pragma unroll 1
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        if (kt + 1 < tiles_k) {
            fp8_pair_block32_weight_scales<Schedule>(operands.weight_block_scales, k, row_begin,
                                                     kt + 1, warp, lane, next_weight_scales);
        }
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        unsigned scale_a[Schedule::kMmaTokens];
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            scale_a[mma_token] = x_scales[stage * BT + scale_token + mma_token * 16];
        }
        mxfp8_mma_a8_compute_stage<Schedule>(activation + stage * BT * BK,
                                             weight + stage * BR * BK, scale_a, weight_scales,
                                             accumulators, warp, lane,
                                             std::make_integer_sequence<int, Schedule::kMmaK>{});
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            weight_scales[mma_row] = next_weight_scales[mma_row];
        }
    }

    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8UnscaledRows{},
                                              mxfp8_a8_shared, accumulators, nullptr,
                                              static_cast<const __nv_bfloat16*>(nullptr), row_begin,
                                              token_begin, operands.rows, token_offset + count, warp,
                                              lane);
}

template <class Schedule, class Output>
void launch_mxfp8_a8_tma_mma(const Mxfp8A8Operands& p, Output output, cudaStream_t stream) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || !aligned(p.x, 16) || !aligned(p.codes, 16) ||
        !aligned(p.x_block_scales, 16) || !aligned(p.weight_block_scales, 16) || p.rows % 16 != 0)
        throw std::invalid_argument(
            "MXFP8 A8 requires positive N/K/T, aligned activations, codes, and block scales");
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("MXFP8 A8 template static K does not match operands");
    }
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("MXFP8 A8 requires complete row/K tiles");
    const int k_tiles = p.k / Schedule::kBlockK;
    const int stride  = fp8_a8_pair_block32_scale_stride(p.tokens);
    const Mxfp8A8TmaDescriptors descriptors{
        fp8_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
        fp8_tma_scale_map(p.x_block_scales, p.tokens, k_tiles, stride, Schedule::kBlockTokens)};
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>,
                                                 const auto& parameter) {
        for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel =
                    mxfp8_a8_tma_mma_kernel<Schedule, Full, Output, InPlace>;
                constexpr int bytes =
                    fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes;
                const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
                kernel<<<blocks, Schedule::kThreads, dynamic, stream>>>(parameter, p, output,
                                                                        offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    };
#ifdef _WIN32
    with_tma_descriptor_parameter(mxfp8_a8_descriptor_staging(), descriptors, stream, launch_slices);
#else
    launch_slices(std::false_type{}, descriptors);
#endif
}

template <class Geometry, class Schedule, class Output>
void mxfp8_a8_gemm(const Tensor& x, const Weight& w, Output output, Fp8A8Block32Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_block32_quantize(x, w, scratch, stream);
    const Mxfp8A8Operands operands{
        scratch.codes,
        scratch.block_scales,
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales),
        w.n,
        w.k,
        x.ne[1]};
    launch_mxfp8_a8_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(operands, output,
                                                                               stream);
}

} // namespace infernix::ops::detail
