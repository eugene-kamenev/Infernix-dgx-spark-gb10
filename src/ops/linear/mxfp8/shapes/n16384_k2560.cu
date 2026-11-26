// Qwen3.8-Flash-Next GDN query/key/value/z projection, MXFP8 with BF16 activations (W8A16, NVIDIA
// FP4-Local). Every width runs the Tensor Core route, whose schedules accumulate a column in one
// order, so a column has the same bits at every T and with either producer (mxfp8_a16_tma_mma.cuh).
// The selection is bench/ops/mxfp8_flash_next_sweep.cu's (GB10, 2026-10-29, 21 repetitions per point,
// tolerance 2 %); all 41 schedules were bit-identical to the reference at T = 1 and T = 300 (with the
// FP64 screen), so the retune changes no output bit. On GB10 (48 SMs, 24 MiB L2, a 48 MiB cold-L2 flush)
// it is 1.041x in geomean over the 28 measured widths and up to 1.89x (T = 8192: 24,875 -> 13,156 us),
// with no width's median slower than the RTX 5090 selection it replaces. T <= 16 keeps the 5090's
// 16x64 tile: at those widths every schedule is within 2 % of the 185 us weight-read floor, and the
// sweep's argmin there bought 0.6 % of geomean while regressing p95 by up to 1.51x at T = 7..9.
// gdn_input_proj runs the same selection into the q/k/v and z destinations, so its rows hold exactly
// linear's bits.
#include "ops/linear/common/output.cuh"
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/linear/mxfp8/mxfp8_shapes.h"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/mxfp8/mxfp8_dispatch.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 2560>;
// The GB10 decode winner: 32 token columns x 16 rows over 4 row warps and 2 stages.
using RingT32R16 = Fp8A16RingMmaSchedule<32, 16, 1, 2, 4, 2>;

template <class Output>
void select(const Mxfp8Operands& p, Output output, cudaStream_t stream) {
    const int tokens = p.tokens;
    if (tokens <= 16) return mxfp8_tma<Geometry, Mxfp8TmaT16R64>(p, output, stream);
    if (tokens <= 32) return mxfp8_tma<Geometry, RingT32R16>(p, output, stream);
    if (tokens <= 64) return mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT32R32>>(p, output, stream);
    if (tokens <= 128) return mxfp8_tma<Geometry, Mxfp8Tma<Fp8A16RingT32R64S3>>(p, output, stream);
    if (tokens <= 192) return mxfp8_tma<Geometry, Mxfp8P2<Mxfp8TmaT64R256>>(p, output, stream);
    if (tokens <= 3072) return mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT128R128G8>>(p, output, stream);
    mxfp8_tma<Geometry, Mxfp8Tma<Mxfp8TmaT64R256G8>>(p, output, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    select(mxfp8_operands(x, weight), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
}

// f16-accumulate-prefill: the GB10 tiles from select at prefill widths (T >= kMxfp8F16MinTokens = 256).
template <class Output>
void select_f16(const Tensor& x, const Weight& weight, Output output, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    if (x.ne[1] <= 3072) return mxfp8_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT128R128>>(x, weight, output, scratch, stream);
    if (x.ne[1] <= 4096) return mxfp8_f16acc<Geometry, Mxfp8Tma<Mxfp8TmaT64R256>>(x, weight, output, scratch, stream);
    mxfp8_f16acc<Geometry, Mxfp8P4<Mxfp8TmaT64R256>>(x, weight, output, scratch, stream);
}

void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    select_f16(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, scratch, stream);
}

// fp8-a8-pair-block32: the activation as two E4M3 terms with per-32 UE8M0 scales, the weight's own group
// scales in the same block-scaled MMA (fp8_a8_pair_tma_mma.cuh); recipe C's pair tiles, not yet swept here.
void launch_pair_block32(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8PairBlock32Workspace scratch,
                         cudaStream_t stream) {
    fp8_a8_pair_block32_gemm<Geometry, true>(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                             scratch, stream);
}
} // namespace

const Mxfp8LinearShape kMxfp8N16384K2560{16384, 2560, launch_a16, launch_f16, kMxfp8F16MinTokens, launch_pair_block32, kMxfp8PairMinTokensN16384};

void mxfp8_gdn_input_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, LinearRoutes routes,
                                  WorkspaceArena* workspace, cudaStream_t stream) {
    const LinearBf16SegmentedOutput<10240, 6144> output{
        {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}};
    if (has_route(routes, LinearRoutes::F16Accumulate) && x.ne[1] >= kMxfp8F16MinTokens) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 gdn_input_proj: the f16-accumulate route requires caller workspace");
        }
        auto scope = workspace->scope();
        return select_f16(x, weight, output, allocate_f16_accumulate_workspace(*workspace, x.ne[1], weight.k),
                          stream);
    }
    if (has_route(routes, LinearRoutes::Fp8A8PairBlock32) && x.ne[1] >= kMxfp8PairMinTokensN16384) {
        if (workspace == nullptr) {
            throw std::invalid_argument("mxfp8 gdn_input_proj: the block32 pair route requires caller workspace");
        }
        auto scope = workspace->scope();
        return fp8_a8_pair_block32_gemm<Geometry, true>(
            x, weight, output, allocate_fp8_a8_pair_block32_workspace(*workspace, x.ne[1], weight.k), stream);
    }
    select(mxfp8_operands(x, weight), output, stream);
}
} // namespace infernix::ops::detail
