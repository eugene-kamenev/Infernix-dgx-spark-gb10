// Qwen3.8-Flash-Next shared-expert gate/up projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (GB10, 2026-10-29, 21 repetitions per point,
// tolerance 2 %); all 41 schedules were bit-identical to the reference at T = 1 and T = 300 (with the
// FP64 screen), so the retune changes no output bit. On GB10 (48 SMs, 24 MiB L2, a 48 MiB cold-L2 flush)
// it is 1.100x in geomean over the 28 measured widths and up to 1.79x (T = 256: 50.91 -> 28.38 us),
// with no width's median or p95 worse than the RTX 5090 selection it replaces. The wins are 49-128
// columns (T = 64: 28.35 -> 20.16 us, T = 96: 29.92 -> 22.24 us with its p95 63.36 -> 22.56 us,
// T = 128: 32.32 -> 22.27 us) and 257-768 (T = 384: 55.01 -> 48.83 us). At 1..48 columns the 5090's
// tile is kept: the sweep's argmin there was worth under 1 % of median and regressed p95 up to 1.92x.
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<1280, 2560>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 48) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT16R32>>(x, weight, out, stream);
    if (tokens <= 64) return mxfp8_linear_a16_tma<Geometry, Mxfp8P2<Mxfp8TmaT32R32>>(x, weight, out, stream);
    if (tokens <= 128) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT32R64S3>>(x, weight, out, stream);
    if (tokens <= 256) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, out, stream);
    if (tokens <= 512) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(x, weight, out, stream);
    if (tokens <= 768) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R64G8>>(x, weight, out, stream);
    if (tokens <= 2048) return mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(x, weight, out, stream);
    mxfp8_linear_a16_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R256G8>>(x, weight, out, stream);
}

// f16-accumulate-prefill: the GB10 TMA tiles from launch_a16 at prefill widths (T >= kMxfp8F16MinTokens = 256).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 256) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Fp8A16RingT64R128>>(x, weight, out, scratch, stream);
    if (tokens <= 512) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, out, scratch, stream);
    if (tokens <= 768) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT64R64>>(x, weight, out, scratch, stream);
    if (tokens <= 2048) return mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, out, scratch, stream);
    mxfp8_linear_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT64R256>>(x, weight, out, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N1280K2560{1280, 2560, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN1280};
} // namespace infernix::ops::detail
