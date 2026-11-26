#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/f16acc/f16acc_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

using Mxfp8Launch     = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Mxfp8F16Launch  = void (*)(const Tensor&, const Weight&, Tensor&, F16AccumulateWorkspace, cudaStream_t);
using Mxfp8PairLaunch = void (*)(const Tensor&, const Weight&, Tensor&, Fp8A8PairBlock32Workspace, cudaStream_t);
using Mxfp8A8Launch   = void (*)(const Tensor&, const Weight&, Tensor&, Fp8A8Block32Workspace, cudaStream_t);

// One registered MXFP8 problem: its exact (A16) route for every width, column-invariant at every T,
// the f16-accumulate-prefill route (LinearRoutes::F16Accumulate, ops/linear/f16acc/f16acc_plan.h)
// from f16_min_tokens, fp8-a8-pair-block32's (LinearRoutes::Fp8A8PairBlock32) from pair_min_tokens,
// and single-term mxfp8-a8's (LinearRoutes::Mxfp8A8) from a8_min_tokens.
struct Mxfp8LinearShape {
    std::int32_t n, k;
    Mxfp8Launch a16;
    Mxfp8F16Launch f16           = nullptr;
    std::int32_t f16_min_tokens  = 0;
    Mxfp8PairLaunch pair_block32 = nullptr;
    std::int32_t pair_min_tokens = 0;
    Mxfp8A8Launch a8             = nullptr;
    std::int32_t a8_min_tokens   = 0;
};

// The f16-accumulate route's admission (prefill widths). Timed on GB10 on 2026-10-29 against the retuned
// exact route (48 MiB cold-L2, 21 reps; docs/gb10/20261029-changelog.md): this one width is wrong for the
// family — the route is 5.3-36.2 % slower from 256 columns on four of the five problems and faster only on
// [13952,2560] from 4096 — so no configuration registers the fast path on GB10
// (qwen4_exp/program/fast_paths.cpp). Each problem's own admission is the field to fix, per measurement.
inline constexpr std::int32_t kMxfp8F16MinTokens = 256;

// The block32 pair's admission per problem: recipe C's measured column-scaled pair admissions for the
// same [N,K] (fp8/shapes, kPairMinTokens; [13952,2560] takes [13312,2560]'s). Timed on GB10 on 2026-10-29
// (same method, opt-in --a8-activations route): slower than the retuned exact route at every width measured
// above its admission on all five problems except [13952,2560] at 4096 columns (-4.9 %) and 8192 (-0.1 %),
// and by up to +87.1 % on [2560,6144] at 8192. Left as recipe C's values pending the A8 campaign
// (docs/tasks/20261029-gb10-post-rebase-perf-tasks.md T2), which retunes these tiles for 48 SMs.
inline constexpr std::int32_t kMxfp8PairMinTokensN16384 = 33;
inline constexpr std::int32_t kMxfp8PairMinTokensN13952 = 97;
inline constexpr std::int32_t kMxfp8PairMinTokensN2560  = 513;
inline constexpr std::int32_t kMxfp8PairMinTokensN1280  = 1025;

// Single-term block-scaled MXFP8 A8 (LinearRoutes::Mxfp8A8) prefill admissions per shape
// (measured on GB10 via infernix_mxfp8_flash_next_sweep --routes-compare): beats exact by
// 13.2-36.6% across [16384,2560], [13952,2560], [2560,6144], and [2560,640] from T >= 256,
// while [1280,2560] keeps exact (small N has too few row tiles to beat exact).
inline constexpr std::int32_t kMxfp8A8MinTokensN16384 = 256;
inline constexpr std::int32_t kMxfp8A8MinTokensN13952 = 256;
inline constexpr std::int32_t kMxfp8A8MinTokensN2560  = 256;
inline constexpr std::int32_t kMxfp8A8MinTokensN1280  = 0x7fffffff;
inline constexpr std::int32_t kMxfp8A8MinTokensN2560K640 = 256;

// Qwen3.8-Flash-Next (NVIDIA FP4-Local, W8A16): the GDN q/k/v/z, QSA query/gate/key/value with the
// indexer, GDN and QSA output and shared-expert gate/up and down projections.
extern const Mxfp8LinearShape kMxfp8N16384K2560;
extern const Mxfp8LinearShape kMxfp8N13952K2560;
extern const Mxfp8LinearShape kMxfp8N2560K6144;
extern const Mxfp8LinearShape kMxfp8N1280K2560;
extern const Mxfp8LinearShape kMxfp8N2560K640;

} // namespace infernix::ops::detail
