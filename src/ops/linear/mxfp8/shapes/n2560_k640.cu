// Qwen3.8-Flash-Next shared expert down projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (GB10, 2026-10-29, 21 repetitions per point,
// tolerance 2 %); all 41 schedules were bit-identical to the reference at T = 1 and T = 300 (with the
// FP64 screen), so the retune changes no output bit. On GB10 (48 SMs, 24 MiB L2, a 48 MiB cold-L2
// flush) it is 1.153x in geomean over the 23 measured widths and up to 1.58x (T = 128: 22.21 -> 14.02
// us), with no width's median slower than the RTX 5090 selection it replaces. The sweep's raw per-width
// argmin was smoothed where it differed by under 6 % of median but regressed p95 (T = 2's 2-producer
// tile: 10.53 us median but a 20.35 us p95 against 12.03 us for the tile kept here); the only width
// whose p95 is worse than before is T = 8192 (564.03 vs 521.09 us) where its median drops 16.3 %.
// Both producers win here — the cp.async tiles at T <= 6, 33..64 and 129..256, the TMA tiles elsewhere
// — while the 5090 selection ran cp.async at every width and reported TMA as never faster.
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 640>;
using RingT32R32S3 = Fp8A16RingMmaSchedule<32, 32, 1, 4, 3, 2>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 6) return mxfp8_linear_a16_tma<Geometry, Mxfp8TmaT32R32>(x, weight, out, stream);
    if (tokens <= 16) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT16R32>>(x, weight, out, stream);
    if (tokens <= 32) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<RingT32R32S3>>(x, weight, out, stream);
    if (tokens <= 64) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Fp8A16RingT32R64S3>>(x, weight, out, stream);
    if (tokens <= 96) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, out, stream);
    if (tokens <= 256) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(x, weight, out, stream);
    if (tokens <= 1024) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64G8>>(x, weight, out, stream);
    mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(x, weight, out, stream);
}

// f16-accumulate-prefill: the GB10 tiles from launch_a16 at prefill widths (T >= kMxfp8F16MinTokens = 256).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 256) return mxfp8_linear_f16acc<Geometry, Mxfp8P2<Mxfp8TmaT64R64>>(x, weight, out, scratch, stream);
    if (tokens <= 1024) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT64R64>>(x, weight, out, scratch, stream);
    mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, out, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N2560K640{2560, 640, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN2560};
} // namespace infernix::ops::detail
