// Qwen3.8-Flash-Next GDN and QSA output projections, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (GB10, 2026-10-29, 21 repetitions per point,
// tolerance 2 %); all 41 schedules were bit-identical to the reference at T = 1 and T = 300 (with the
// FP64 screen), so the retune changes no output bit. On GB10 (48 SMs, 24 MiB L2, a 48 MiB cold-L2 flush)
// it is 1.096x in geomean over the 28 measured widths and up to 1.90x (T = 8192: 9,134 -> 4,805 us),
// with no width's median slower than the RTX 5090 selection it replaces. The 16x16 decode tile is the
// defect this closes: at T = 1..8 it took 82.7-88.0 us against 75.6-77.6 us for the 16x64 tile with two
// producer warps. T = 9..32 keeps the 5090's TMA tile, which is already within 1.5 % of the sweep's best
// there. The widths whose p95 is worse are T = 2 (1.27x), T = 3 (1.42x), T = 512 (1.25x) and
// T = 1536 (1.13x); these cold-L2 p95s are bimodal (a schedule reads either ~77 us or ~100-170 us) and
// the mode moves between schedules and widths with no consistent pattern, so the medians drove the
// choice and the tails are reported, not optimized.
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/mxfp8/mxfp8_a8_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 6144>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 8) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R64>>(x, weight, out, stream);
    if (tokens <= 32) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT16R32>>(x, weight, out, stream);
    if (tokens <= 64) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT32R64S3>>(x, weight, out, stream);
    if (tokens <= 128) return mxfp8_linear_a16_tma<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    if (tokens <= 512) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64G8>>(x, weight, out, stream);
    if (tokens <= 2048) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(x, weight, out, stream);
    mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R256G8>>(x, weight, out, stream);
}

// f16-accumulate-prefill: the GB10 tiles from launch_a16 at prefill widths (T >= kMxfp8F16MinTokens = 256).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 512) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT64R64G8>>(x, weight, out, scratch, stream);
    if (tokens <= 2048) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(x, weight, out, scratch, stream);
    mxfp8_linear_f16acc<Geometry, Mxfp8P4<Mxfp8TmaT64R256>>(x, weight, out, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Block32Workspace scratch,
               cudaStream_t stream) {
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    if (x.ne[1] <= 512) return mxfp8_a8_gemm<Geometry, Mxfp8A8T64R64G8S3>(x, weight, output, scratch, stream);
    if (x.ne[1] <= 2048) return mxfp8_a8_gemm<Geometry, Mxfp8A8T128R128G8S3>(x, weight, output, scratch, stream);
    mxfp8_a8_gemm<Geometry, Mxfp8A8T64R256G8S2>(x, weight, output, scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N2560K6144{
    2560, 6144, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN2560,
    launch_a8, kMxfp8A8MinTokensN2560};
} // namespace infernix::ops::detail
