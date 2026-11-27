#pragma once

// BF16 matrices as 4-bit weights (LOSSY: opt-in --w4-weights): NVFP4 RowSplit in a 3-plane layout
// (E2M1 nibbles [n, padded_k / 2] with padded_k = k rounded up to 128, FP32 row_scale [n] in qhigh,
// and E4M3FN group-16 scales [n, padded_k / 16]). For row r and group g of 16 elements:
//   row_scale[r] = max_j |W[r, j]| / (6 * 448)
//   scale[r, g]  = e4m3(max_{blk} |W| / (6 * row_scale[r]))
//   code         = e2m1(W / (e4m3_val(scale[r, g]) * row_scale[r]))
// Used by Inkling 2's --w4-weights.

#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

// Whether `w` is a matrix quantize_bf16_w4_g16 takes: BF16 Contiguous [n, k], k a multiple of 16, 16-byte aligned.
[[nodiscard]] bool w4_g16_quantizable(const Weight& w) noexcept;

// A row's code elements (padded up to 128). Code bytes per row = padded_k / 2.
[[nodiscard]] std::int32_t w4_g16_padded_columns(std::int32_t k) noexcept;

// Writes the W4 form of `src` on `stream`: E2M1 nibbles to `codes` (n * padded_k / 2 bytes, 16-byte aligned),
// FP32 per-row multipliers to `row_scales` (n floats, 4-byte aligned), and E4M3FN scales to `scales`
// (n * padded_k / 16 bytes, 8-byte aligned), none overlapping src. Returns the Weight describing them.
[[nodiscard]] Weight quantize_bf16_w4_g16(const Weight& src, void* codes, void* row_scales, void* scales,
                                          cudaStream_t stream);

} // namespace infernix::ops
