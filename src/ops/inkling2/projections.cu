// Inkling 2's few-input projections in one launch (include/infernix/ops/inkling2_projections.h): the BF16 skinny
// GEMV's arithmetic (ops/linear/bf16/bf16_skinny.cuh) over several matrices, each row group of a CTA in one segment.
// A warp owns RowsPerWarp rows; lane l folds chunks l, l + 32, ... in order, kAhead chunks of each row in flight
// (the first kAhead loaded before the PDL wait), every column of the pass from one load of the chunk.

#include "infernix/ops/inkling2_projections.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::inkling2 {
namespace {

constexpr int kAhead   = 4;
constexpr int kMaxPass = 16;

// Compact segment descriptor matching original ProjectionSegment size so Params stays in constant memory without
// inflating register/stack pressure on tight 128-reg wide kernels. For Storage::W4, scales holds w4_scales, escapes
// holds w4_row_scales, and stride holds w4_stride.
struct Seg {
    const std::uint16_t* weight = nullptr;
    std::int32_t rows           = 0;
    std::int32_t input          = 0;
    void* out                   = nullptr;
    const void* codes           = nullptr;
    const void* nibbles         = nullptr;
    const unsigned* words       = nullptr;
    const void* escapes         = nullptr;
    const void* scales          = nullptr;
    std::int32_t stride         = 0;
};

struct Params {
    const std::uint16_t* x[3];
    Seg seg[kMaxProjectionSegments];
    std::int32_t cta_begin[kMaxProjectionSegments + 1];
    std::int32_t count;
    std::int32_t k;
    std::int32_t c0;   // the pass's first column
    std::int32_t cols; // its columns (<= Tokens; a wide launch: the call's)
    std::int32_t block_columns; // a wide launch: the columns each CTA row (grid y) takes, a multiple of kMaxPass
};

enum class Storage { BF16, P12, Q8, W4 };

// A chunk of 8 weights as stored: BF16 words, bf16_p12 code bytes and exponent nibbles, Q8 code bytes and FP16 scale,
// or W4 E2M1 code bytes and E4M3FN group-16 scale.
struct P12Raw {
    uint2 code;
    unsigned nib;
};
struct Q8Raw {
    uint2 code;
    unsigned short scale;
};
struct W4Raw {
    unsigned code;
    std::uint8_t scale;
};

// Row `row`'s chunk `chunk` of a Q8 segment (stride code bytes a row, a scale per 4 chunks).
__device__ __forceinline__ Q8Raw q8_load(const Seg& seg, int row, int chunk) {
    const std::size_t codes = static_cast<std::size_t>(row) * (seg.stride / 8) + chunk;
    const std::size_t scale = static_cast<std::size_t>(row) * (seg.stride / 32) + chunk / 4;
    return Q8Raw{__ldcs(static_cast<const uint2*>(seg.codes) + codes),
                 __ldcs(static_cast<const unsigned short*>(seg.scales) + scale)};
}

// Row `row`'s chunk `chunk` (8 elements = 4 code bytes) of a W4 segment (stride elements a row, a scale per 2 chunks).
__device__ __forceinline__ W4Raw w4_load(const Seg& seg, int row, int chunk) {
    const std::size_t codes = static_cast<std::size_t>(row) * (seg.stride / 8) + chunk;
    const std::size_t scale = static_cast<std::size_t>(row) * (seg.stride / 16) + chunk / 2;
    return W4Raw{__ldcs(static_cast<const unsigned*>(seg.codes) + codes),
                 __ldcs(static_cast<const std::uint8_t*>(seg.scales) + scale)};
}

// A Q8 chunk's 8 weights, code * scale (exact in FP32: 8 by 11 significant bits). Each code byte c becomes the float
// 2^23 + (c + 128) by a byte permute into 0x4B000000, less 2^23 + 128 (exact).
__device__ __forceinline__ void q8_decode(const Q8Raw& raw, float (&w)[8]) {
    const float s     = __half2float(__ushort_as_half(raw.scale));
    const unsigned lo = raw.code.x ^ 0x80808080U, hi = raw.code.y ^ 0x80808080U;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j]     = (__uint_as_float(__byte_perm(lo, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
        w[4 + j] = (__uint_as_float(__byte_perm(hi, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
    }
}

// A W4 chunk's 8 weights, E2M1 * E4M3 (exact in FP16 -> float; per-row FP32 multiplier applies once at epilogue).
__device__ __forceinline__ void w4_decode(const W4Raw& raw, float (&w)[8]) {
    const __half2 mult = ::infernix::ops::detail::decode_nvfp4_e4m3_f16x2(raw.scale);
    unsigned h0, h1, h2, h3;
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
        : "r"(raw.code));
    const unsigned hs[4] = {h0, h1, h2, h3};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        union {
            unsigned bits;
            __half2 h2;
        } in{hs[i]};
        const float2 f = __half22float2(__hmul2(in.h2, mult));
        w[2 * i]       = f.x;
        w[2 * i + 1]   = f.y;
    }
}

template <int Tokens, int Warps, int RowsPerWarp, Storage S, class Out>
__global__ void __launch_bounds__(Warps * 32) projections_kernel(const Params p) {
    constexpr bool P12 = S == Storage::P12;
    using Raw = std::conditional_t<
        S == Storage::W4, W4Raw,
        std::conditional_t<S == Storage::Q8, Q8Raw, std::conditional_t<P12, P12Raw, uint4>>>;
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta  = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const Seg seg = p.seg[s];
    const int first             = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks            = p.k / 8;
    std::size_t base[RowsPerWarp];
    int rows[RowsPerWarp];
    ::infernix::ops::detail::P12Row prow[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = min(first + r, seg.rows - 1);
        rows[r]       = row;
        base[r]       = static_cast<std::size_t>(row) * chunks;
        if constexpr (P12) { prow[r] = ::infernix::ops::detail::p12_row(seg.words, row); }
    }
    (void)rows, (void)base;
    const auto load = [&](int r, int chunk) {
        if constexpr (S == Storage::W4) {
            return w4_load(seg, rows[r], chunk);
        } else if constexpr (S == Storage::Q8) {
            return q8_load(seg, rows[r], chunk);
        } else if constexpr (P12) {
            return P12Raw{__ldcs(static_cast<const uint2*>(seg.codes) + base[r] + chunk),
                          __ldcs(static_cast<const unsigned*>(seg.nibbles) + base[r] + chunk)};
        } else {
            return __ldcs(reinterpret_cast<const uint4*>(seg.weight) + base[r] + chunk);
        }
    };
    const auto decode = [&](const Raw& raw, int r, int chunk) {
        if constexpr (P12) {
            return ::infernix::ops::detail::p12_decode_chunk(raw.code, raw.nib, prow[r], chunk,
                                                             static_cast<const unsigned*>(seg.escapes));
        } else {
            (void)r, (void)chunk;
            return raw;
        }
    };
    Raw w[kAhead][RowsPerWarp];
#pragma unroll
    for (int a = 0; a < kAhead; ++a) {
        const int chunk = lane + 32 * a;
        if (chunk < chunks) {
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = load(r, chunk); }
        }
    }
    pdl::enter_streaming();

    const std::uint16_t* x = p.x[seg.input] + static_cast<std::size_t>(p.c0) * p.k;
    float sums[RowsPerWarp][Tokens] = {};
    for (int start = 0; start < chunks; start += 32 * kAhead) {
#pragma unroll
        for (int a = 0; a < kAhead; ++a) {
            const int chunk = start + lane + 32 * a;
            if (chunk >= chunks) { break; }
            if constexpr (S == Storage::Q8 || S == Storage::W4) {
                float wq[RowsPerWarp][8];
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    if constexpr (S == Storage::W4) {
                        w4_decode(w[a][r], wq[r]);
                    } else {
                        q8_decode(w[a][r], wq[r]);
                    }
                }
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    if (t >= p.cols) { break; }
                    const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * p.k) + chunk);
                    const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float2 xf = __bfloat1622float2(xv[i]);
                            sums[r][t]      = fmaf(wq[r][2 * i], xf.x, sums[r][t]);
                            sums[r][t]      = fmaf(wq[r][2 * i + 1], xf.y, sums[r][t]);
                        }
                    }
                }
            } else {
                uint4 wd[RowsPerWarp];
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) { wd[r] = decode(w[a][r], r, chunk); }
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    if (t >= p.cols) { break; }
                    const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * p.k) + chunk);
                    const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r) {
                        const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&wd[r]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float2 wf = __bfloat1622float2(wv[i]);
                            const float2 xf = __bfloat1622float2(xv[i]);
                            sums[r][t]      = fmaf(wf.x, xf.x, sums[r][t]);
                            sums[r][t]      = fmaf(wf.y, xf.y, sums[r][t]);
                        }
                    }
                }
            }
            const int next = chunk + 32 * kAhead;
            if (next < chunks) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) { w[a][r] = load(r, next); }
            }
        }
    }
    pdl::trigger_dependents();
    Out* out = static_cast<Out*>(seg.out) + static_cast<std::size_t>(p.c0) * seg.rows;
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t >= p.cols) { break; }
            float v = sums[r][t];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && first + r < seg.rows) {
                if constexpr (S == Storage::W4) { v *= static_cast<const float*>(seg.escapes)[rows[r]]; }
                if constexpr (std::is_same_v<Out, float>) {
                    out[static_cast<std::size_t>(t) * seg.rows + first + r] = v;
                } else {
                    out[static_cast<std::size_t>(t) * seg.rows + first + r] = __float2bfloat16_rn(v);
                }
            }
        }
    }
}

// A wide call (more than kMaxPass columns, BF16 storage): each lane holds every chunk of its warp's rows in registers
// (Chunks a row), loaded once before the PDL wait, and the launch runs the call's passes of kMaxPass columns itself.
// A (row, column)'s fold is projections_kernel's (chunks l, l + 32, ... in order, 8 FMAs in k order a chunk, the same
// butterfly), so are its bits; only the weights' reuse across passes differs.
// The small instance (the gates' downs) at most 128 registers a thread, eight 2-warp CTAs an SM (measured faster:
// 128 -> 97 ms of a 2,048-column call's gates); the large one unbounded (bounding it spilled W_o's and left K2's
// time unchanged).
template <Storage S, class Held>
__device__ __forceinline__ Held wide_load_chunk(const Seg& seg, int row, int chunk, int chunks) {
    if constexpr (S == Storage::W4) {
        return chunk < chunks ? w4_load(seg, row, chunk) : W4Raw{};
    } else if constexpr (S == Storage::Q8) {
        return chunk < chunks ? q8_load(seg, row, chunk) : Q8Raw{};
    } else {
        return chunk < chunks
                   ? __ldcs(reinterpret_cast<const uint4*>(seg.weight) + static_cast<std::size_t>(row) * chunks + chunk)
                   : make_uint4(0U, 0U, 0U, 0U);
    }
}

template <Storage S, int RowsPerWarp, class Held>
__device__ __forceinline__ void wide_decode_rows(const Held (&held_row)[RowsPerWarp], float2 (&wf)[RowsPerWarp][4]) {
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        if constexpr (S == Storage::W4) {
            W4Raw held = held_row[r];
            float wq[8];
            w4_decode(held, wq);
#pragma unroll
            for (int i = 0; i < 4; ++i) { wf[r][i] = make_float2(wq[2 * i], wq[2 * i + 1]); }
        } else if constexpr (S == Storage::Q8) {
            // The decode stays in the pass loop: hoisted, every chunk's decoded floats spilled (W_o's K: 169 stores a thread).
            Q8Raw held = held_row[r];
            asm volatile("" : "+r"(held.code.x), "+r"(held.code.y), "+h"(held.scale));
            float wq[8];
            q8_decode(held, wq);
#pragma unroll
            for (int i = 0; i < 4; ++i) { wf[r][i] = make_float2(wq[2 * i], wq[2 * i + 1]); }
        } else {
            const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&held_row[r]);
#pragma unroll
            for (int i = 0; i < 4; ++i) { wf[r][i] = __bfloat1622float2(wv[i]); }
        }
    }
}

template <int Chunks, int Warps, int RowsPerWarp, Storage S, class Held, class Out>
__device__ __forceinline__ void wide_run_passes(const Seg& seg, const std::uint16_t* x, int k, int cols,
                                                int block_columns, int first, int chunks, int lane,
                                                const Held (&w)[Chunks][RowsPerWarp]) {
    Out* out       = static_cast<Out*>(seg.out);
    const int from = static_cast<int>(blockIdx.y) * block_columns, to = min(cols, from + block_columns);
    for (int c0 = from; c0 < to; c0 += kMaxPass) {
        const int n = min(kMaxPass, to - c0);
        float sums[RowsPerWarp][kMaxPass] = {};
#pragma unroll
        for (int c = 0; c < Chunks; ++c) {
            const int chunk = lane + 32 * c;
            if (chunk >= chunks) { break; }
            float2 wf[RowsPerWarp][4];
            wide_decode_rows<S, RowsPerWarp>(w[c], wf);
#pragma unroll
            for (int t = 0; t < kMaxPass; ++t) {
                if (t >= n) { break; }
                const uint4 xp = *(reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(c0 + t) * k) + chunk);
                const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
                float2 xf[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { xf[i] = __bfloat1622float2(xv[i]); }
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sums[r][t] = fmaf(wf[r][i].x, xf[i].x, sums[r][t]);
                        sums[r][t] = fmaf(wf[r][i].y, xf[i].y, sums[r][t]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int t = 0; t < kMaxPass; ++t) {
                if (t >= n) { break; }
                float v = sums[r][t];
                for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
                if (lane == 0 && first + r < seg.rows) {
                    if constexpr (S == Storage::W4) { v *= static_cast<const float*>(seg.escapes)[first + r]; }
                    const std::size_t at = static_cast<std::size_t>(c0 + t) * seg.rows + first + r;
                    if constexpr (std::is_same_v<Out, float>) {
                        out[at] = v;
                    } else {
                        out[at] = __float2bfloat16_rn(v);
                    }
                }
            }
        }
    }
}

// Keep the legacy <bool Q8> kernel symbol name unchanged for BF16/Q8 stack accounting.
template <int Chunks, int Warps, int RowsPerWarp, bool Q8, class Out>
__global__ void __launch_bounds__(Warps * 32, Warps == 2 ? 8 : 1) projections_wide_kernel(const Params p) {
    constexpr Storage S = Q8 ? Storage::Q8 : Storage::BF16;
    using Held          = std::conditional_t<Q8, Q8Raw, uint4>;
    const int lane      = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta       = static_cast<int>(blockIdx.x);
    int s               = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const Seg seg    = p.seg[s];
    const int first  = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks = p.k / 8;
    Held w[Chunks][RowsPerWarp];
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
        const int chunk = lane + 32 * c;
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            w[c][r] = wide_load_chunk<S, Held>(seg, min(first + r, seg.rows - 1), chunk, chunks);
        }
    }
    pdl::enter_streaming();
    wide_run_passes<Chunks, Warps, RowsPerWarp, S, Held, Out>(seg, p.x[seg.input], p.k, p.cols, p.block_columns, first,
                                                              chunks, lane, w);
    pdl::trigger_dependents();
}

template <int Chunks, int Warps, int RowsPerWarp, class Out>
__global__ void __launch_bounds__(Warps * 32, Warps == 2 ? 8 : 1) projections_wide_w4_kernel(const Params p) {
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int cta  = static_cast<int>(blockIdx.x);
    int s          = 0;
    while (s + 1 < p.count && cta >= p.cta_begin[s + 1]) { ++s; }
    const Seg seg    = p.seg[s];
    const int first  = ((cta - p.cta_begin[s]) * Warps + warp) * RowsPerWarp;
    const int chunks = p.k / 8;
    W4Raw w[Chunks][RowsPerWarp];
#pragma unroll
    for (int c = 0; c < Chunks; ++c) {
        const int chunk = lane + 32 * c;
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            w[c][r] = wide_load_chunk<Storage::W4, W4Raw>(seg, min(first + r, seg.rows - 1), chunk, chunks);
        }
    }
    pdl::enter_streaming();
    wide_run_passes<Chunks, Warps, RowsPerWarp, Storage::W4, W4Raw, Out>(seg, p.x[seg.input], p.k, p.cols,
                                                                         p.block_columns, first, chunks, lane, w);
    pdl::trigger_dependents();
}

// The wide kernel for K = 8 * 32 * Chunks at most (Chunks of 1, 9, 12 and 18: the forget/gain, K2, W_o and gate-down
// shapes); false for another K (or W4 at Chunks = 18 where registers spill, falling back to projections_kernel).
template <int Warps, int RowsPerWarp, Storage S, class Out>
bool launch_wide(const Params& p, dim3 grid, cudaStream_t stream) {
    const pdl::LaunchConfig config{grid, dim3(Warps * 32), 0, stream};
    if constexpr (S == Storage::W4) {
        switch ((p.k / 8 + 31) / 32) {
        case 1:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_w4_kernel<1, Warps, RowsPerWarp, Out>, p));
            return true;
        case 9:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_w4_kernel<9, Warps, RowsPerWarp, Out>, p));
            return true;
        case 12:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_w4_kernel<12, Warps, RowsPerWarp, Out>, p));
            return true;
        default: return false;
        }
    } else {
        constexpr bool Q8 = S == Storage::Q8;
        switch ((p.k / 8 + 31) / 32) {
        case 1:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<1, Warps, RowsPerWarp, Q8, Out>, p));
            return true;
        case 9:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<9, Warps, RowsPerWarp, Q8, Out>, p));
            return true;
        case 12:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<12, Warps, RowsPerWarp, Q8, Out>, p));
            return true;
        case 18:
            CUDA_CHECK(pdl::launch_consumer(config, projections_wide_kernel<18, Warps, RowsPerWarp, Q8, Out>, p));
            return true;
        default: return false;
        }
    }
}

template <Storage S>
bool launch_wide_storage(const Params& p, bool small, bool fp32, dim3 grid, cudaStream_t stream) {
    return small ? (fp32 ? launch_wide<2, 1, S, float>(p, grid, stream)
                         : launch_wide<2, 1, S, __nv_bfloat16>(p, grid, stream))
                 : (fp32 ? launch_wide<8, 2, S, float>(p, grid, stream)
                         : launch_wide<8, 2, S, __nv_bfloat16>(p, grid, stream));
}

template <int Warps, int RowsPerWarp, Storage S, class Out>
void launch_pass(const Params& p, int tokens, unsigned grid, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(grid), dim3(Warps * 32), 0, stream};
    switch (tokens) {
    case 1: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<1, Warps, RowsPerWarp, S, Out>, p)); break;
    case 2: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<2, Warps, RowsPerWarp, S, Out>, p)); break;
    case 4: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<4, Warps, RowsPerWarp, S, Out>, p)); break;
    case 8: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<8, Warps, RowsPerWarp, S, Out>, p)); break;
    default: CUDA_CHECK(pdl::launch_consumer(config, projections_kernel<16, Warps, RowsPerWarp, S, Out>, p)); break;
    }
}

template <Storage S>
void launch_storage(const Params& p, bool small, bool fp32, int tokens, unsigned grid, cudaStream_t stream) {
    if (small) {
        fp32 ? launch_pass<2, 1, S, float>(p, tokens, grid, stream)
             : launch_pass<2, 1, S, __nv_bfloat16>(p, tokens, grid, stream);
    } else {
        fp32 ? launch_pass<8, 2, S, float>(p, tokens, grid, stream)
             : launch_pass<8, 2, S, __nv_bfloat16>(p, tokens, grid, stream);
    }
}

template <class P>
bool aligned16(const P* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

} // namespace

void projections(const ProjectionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.k <= 0 || a.k % 8 != 0 || a.count <= 0 || a.count > kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 projections: 1-8 segments, positive columns, K a multiple of 8");
    }
    std::int32_t widest = 0;
    const ProjectionSegment& lead = a.segments[0];
    const Storage kind            = lead.codes == nullptr       ? Storage::BF16
                                  : lead.w4_scales != nullptr ? Storage::W4
                                  : lead.q8_scales != nullptr ? Storage::Q8
                                                              : Storage::P12;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        bool storage               = false;
        switch (kind) {
        case Storage::BF16: storage = g.codes == nullptr && g.weight != nullptr && aligned16(g.weight); break;
        case Storage::P12:
            storage = g.codes != nullptr && g.nibbles != nullptr && g.words != nullptr && g.escapes != nullptr &&
                      g.weight == nullptr && g.q8_scales == nullptr && g.w4_scales == nullptr;
            break;
        case Storage::Q8:
            storage = g.codes != nullptr && g.q8_scales != nullptr && g.weight == nullptr && g.nibbles == nullptr &&
                      g.w4_scales == nullptr && g.q8_stride >= a.k && g.q8_stride % 32 == 0 && aligned16(g.codes);
            break;
        case Storage::W4:
            storage = g.codes != nullptr && g.w4_scales != nullptr && g.w4_row_scales != nullptr &&
                      g.weight == nullptr && g.nibbles == nullptr && g.q8_scales == nullptr &&
                      g.w4_stride >= a.k && g.w4_stride % 128 == 0 && aligned16(g.codes);
            break;
        }
        if (!storage || g.rows <= 0 || g.input < 0 || g.input > 2 || a.x[g.input] == nullptr || g.out == nullptr ||
            !aligned16(a.x[g.input])) {
            throw std::invalid_argument("inkling2 projections: a segment's weight (one storage for the call), input or "
                                        "output is missing or misaligned");
        }
        widest = std::max(widest, g.rows);
    }
    // Small matrices only: one row a warp, two warps a CTA (the BF16 skinny's small-N shape); otherwise 2 rows a warp,
    // 8 warps. The arithmetic of a row is the same in both.
    const bool small = widest <= 256;
    const int rows_per_cta = small ? 2 : 16;
    Params p{};
    for (int i = 0; i < 3; ++i) { p.x[i] = a.x[i]; }
    p.count = a.count;
    p.k     = a.k;
    std::int32_t ctas = 0;
    for (int s = 0; s < a.count; ++s) {
        const ProjectionSegment& g = a.segments[s];
        Seg dst{};
        dst.weight     = g.weight;
        dst.rows       = g.rows;
        dst.input      = g.input;
        dst.out        = g.out;
        dst.codes      = g.codes;
        dst.nibbles    = g.nibbles;
        dst.words      = g.words;
        dst.escapes    = kind == Storage::W4 ? static_cast<const void*>(g.w4_row_scales)
                                             : static_cast<const void*>(g.escapes);
        dst.scales     = kind == Storage::W4 ? static_cast<const void*>(g.w4_scales)
                                             : static_cast<const void*>(g.q8_scales);
        dst.stride     = kind == Storage::W4 ? g.w4_stride : g.q8_stride;
        p.seg[s]       = dst;
        p.cta_begin[s] = ctas;
        ctas += (a.segments[s].rows + rows_per_cta - 1) / rows_per_cta;
    }
    p.cta_begin[a.count] = ctas;
    const auto grid      = static_cast<unsigned>(ctas);
    if (kind != Storage::P12 && a.columns > kMaxPass) { // one launch over every pass, the weights held in registers
        // The narrow kernel's CTA shapes (2 rows a warp, 8 warps; one row a warp halved x's reuse and ran K2 1.5x
        // slower); the columns split into blocks (grid y) so the launch has about kWideCtas CTAs whatever its rows: a
        // matrix of few rows (the gates' downs) still fills the GPU; each column stays whole in one CTA.
        constexpr int kWideCtas = 1024;
        const int wide_rows = small ? 2 : 16;
        Params w            = p;
        std::int32_t wctas  = 0;
        for (int s = 0; s < a.count; ++s) {
            w.cta_begin[s] = wctas;
            wctas += (a.segments[s].rows + wide_rows - 1) / wide_rows;
        }
        w.cta_begin[a.count] = wctas;
        const int passes   = (a.columns + kMaxPass - 1) / kMaxPass;
        const int blocks   = std::clamp((kWideCtas + wctas - 1) / wctas, 1, passes);
        w.c0               = 0;
        w.cols             = a.columns;
        w.block_columns    = (passes + blocks - 1) / blocks * kMaxPass;
        const dim3 wgrid(static_cast<unsigned>(wctas),
                         static_cast<unsigned>((a.columns + w.block_columns - 1) / w.block_columns));
        const bool wide = kind == Storage::W4 ? launch_wide_storage<Storage::W4>(w, small, a.fp32_out, wgrid, stream)
                        : kind == Storage::Q8 ? launch_wide_storage<Storage::Q8>(w, small, a.fp32_out, wgrid, stream)
                                              : launch_wide_storage<Storage::BF16>(w, small, a.fp32_out, wgrid, stream);
        if (wide) { return; }
    }
    for (std::int32_t c0 = 0; c0 < a.columns; c0 += kMaxPass) {
        p.c0   = c0;
        p.cols = std::min(kMaxPass, a.columns - c0);
        const int tokens = p.cols == 1 ? 1 : p.cols == 2 ? 2 : p.cols <= 4 ? 4 : p.cols <= 8 ? 8 : 16;
        switch (kind) {
        case Storage::BF16: launch_storage<Storage::BF16>(p, small, a.fp32_out, tokens, grid, stream); break;
        case Storage::P12: launch_storage<Storage::P12>(p, small, a.fp32_out, tokens, grid, stream); break;
        case Storage::Q8: launch_storage<Storage::Q8>(p, small, a.fp32_out, tokens, grid, stream); break;
        case Storage::W4: launch_storage<Storage::W4>(p, small, a.fp32_out, tokens, grid, stream); break;
        }
    }
}

} // namespace infernix::ops::inkling2
