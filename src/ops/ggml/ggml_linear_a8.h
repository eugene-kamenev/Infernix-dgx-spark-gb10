#pragma once

// The 8-bit activation routes of ops::ggml_linear (`--a8-activations`: LOSSY, opt-in;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.31): LinearRoutes::GgmlA8 for calls of 1 to kA8MaxColumns
// columns (ops/ggml/ggml_linear_a8.cuh's kernels) and LinearRoutes::GgmlA8Wide for calls of kA8WideMinColumns or
// more (ops/ggml/ggml_a8_tile.cuh's tiles over decoded planes). Both quantize each column with
// ops/ggml/ggml_a8.h's codec, so a column's codes and slice scales are the same on either route.

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/ggml_linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::ops {

// The widest call of the narrow route (ggml_linear's narrow calls: one below its tiles' kTileMinColumns).
inline constexpr std::int32_t kA8MaxColumns = 31;
// The narrowest call of the wide route (ggml_linear's wide calls, whose K is also a multiple of 64).
// GB10 (48 SMs) retune 2026-10-29: the wide tile is 128 rows x 128 columns, so a 32- or 64-column call
// fills a quarter to a half of it and runs 1.65x-2.25x slower than the exact route's wide A16 tiles on
// every IQ3_S dense shape (gdn_qkvz [16384,2560] T=32: 749.3 us against 343.9 us; T=64: 751.7 against
// 454.6). From 128 columns the int8 tiles over their own planes win everywhere instead (T=128: 747.4
// against 1037.2; T=4096: 5677.9 against 16582.5), so the admission sits at the tile's own width.
inline constexpr std::int32_t kA8WideMinColumns = 128;
// Columns the wide route encodes at once (a call of more runs its tiles batch by batch over the same planes).
inline constexpr std::int32_t kA8WideBatchColumns = 2048;
// Workspace of the wide route's decoded planes (the exact and BF16 wide routes' kPlaneBytes,
// right-sized to 16 MiB for GB10's 24 MiB L2 cache).
inline constexpr std::size_t kA8WidePlaneBytes = 16ULL << 20;

// ggml_linear's arguments, already validated by it (shapes, formats, layouts, outputs, permutation). Each
// column of x (input element j at x[perm(j)]) is encoded once by ops/ggml/ggml_a8.h's codec into
// `workspace` (a8_linear_workspace_bytes: 1.25 bytes per element and a 16-byte rounding per column, under
// the exact route's 3.125 that ggml_linear_workspace_bytes sizes), then every segment's rows take their terms.
void ggml_linear_a8(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                    const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace,
                    cudaStream_t stream);

// The bytes ggml_linear_a8 allocates from its workspace for `columns` columns of K = k.
[[nodiscard]] std::size_t a8_linear_workspace_bytes(std::int32_t k, std::int32_t columns, bool sums);

[[nodiscard]] std::int32_t a8_linear_column_words(std::int32_t k, bool sums);
void a8_encode_columns(const Tensor& x, Tensor& encoded, cudaStream_t stream);

// The wide route (ggml_linear_a8_wide.cu), for a call of kA8WideMinColumns or more columns of K a multiple of 64
// with ggml_linear's validated arguments: the segments' decoded planes (at most kA8WidePlaneBytes) and the encoded
// columns of batches of up to kA8WideBatchColumns.
void ggml_linear_a8_wide(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                         const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace,
                         cudaStream_t stream);

// The bytes ggml_linear_a8_wide allocates from its workspace for `columns` columns of K = k (planes included).
[[nodiscard]] std::size_t a8_wide_linear_workspace_bytes(std::int32_t k, std::int32_t columns);

} // namespace infernix::ops
