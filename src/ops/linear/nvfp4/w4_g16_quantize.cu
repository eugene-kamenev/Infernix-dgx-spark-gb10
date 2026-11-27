// quantize_bf16_w4_g16 (include/infernix/ops/w4_g16.h): a BF16 Contiguous matrix's 4-bit NVFP4 RowSplit form
// (E2M1 codes [n][padded_k / 2], per-row FP32 multipliers [n] in qhigh, E4M3FN group-16 scales [n][padded_k / 16]).

#include "infernix/ops/w4_g16.h"

#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256; // 8 warps = 16 half-warps per CTA (one row per CTA)

[[nodiscard]] int padded_columns(int k) { return (k + 127) / 128 * 128; }

// CTA r takes row r.
// Pass 1: all 256 threads reduce row max -> row_scale = row_max / (6 * 448).
// Pass 2: each half-warp (16 threads) quantizes one 16-element group at a time.
__global__ void __launch_bounds__(kThreads)
    quantize_w4_kernel(const __nv_bfloat16* __restrict__ src, int k, int padded_k,
                       std::uint8_t* __restrict__ codes, float* __restrict__ row_scales,
                       std::uint8_t* __restrict__ scales) {
    __shared__ float warp_max[kThreads / 32];
    __shared__ float s_row_scale;

    const int row  = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    const __nv_bfloat16* row_src = src + static_cast<std::size_t>(row) * k;

    float rmax = 0.0F;
    for (int col = tid; col < k; col += kThreads) { rmax = fmaxf(rmax, fabsf(__bfloat162float(row_src[col]))); }
    for (int o = 16; o > 0; o >>= 1) { rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFFU, rmax, o)); }
    if (lane == 0) { warp_max[warp] = rmax; }
    __syncthreads();
    if (warp == 0) {
        float m = lane < (kThreads / 32) ? warp_max[lane] : 0.0F;
        for (int o = (kThreads / 32) / 2; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
        if (lane == 0) {
            const float rs = m > 0.0F ? (m / (6.0F * 448.0F)) : 0.0F;
            s_row_scale    = rs;
            row_scales[row] = rs;
        }
    }
    __syncthreads();

    const float row_scale = s_row_scale;
    const int groups      = padded_k / 16;
    const int hg          = tid >> 4; // half-warp index [0..15]
    const int hl          = tid & 15; // lane within 16-element group [0..15]
    std::uint8_t* row_codes  = codes + static_cast<std::size_t>(row) * (padded_k / 2);
    std::uint8_t* row_scales_g = scales + static_cast<std::size_t>(row) * groups;

    for (int g = hg; g < groups; g += kThreads / 16) {
        const int col = g * 16 + hl;
        const float v = col < k ? __bfloat162float(row_src[col]) : 0.0F;
        float gm      = fabsf(v);
        for (int o = 8; o > 0; o >>= 1) { gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFFU, gm, o)); }

        std::uint8_t se = 0;
        if (row_scale > 0.0F && gm > 0.0F) {
            se = __nv_cvt_float_to_fp8(__fdiv_rn(gm, 6.0F * row_scale), __NV_SATFINITE, __NV_E4M3);
        }
        const float sv  = se != 0 ? detail::decode_nvfp4_e4m3(se) : 0.0F;
        const float q   = (sv > 0.0F && row_scale > 0.0F) ? __fdiv_rn(v, sv * row_scale) : 0.0F;
        const float q_o = __shfl_xor_sync(0xFFFFFFFFU, q, 1);

        if ((hl & 1) == 0) {
            unsigned packed = 0;
            asm("{\n"
                ".reg .b8 b0;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
                "mov.b32 %0, {b0, b0, b0, b0};\n"
                "}\n"
                : "=r"(packed)
                : "f"(q), "f"(q_o));
            row_codes[g * 8 + (hl >> 1)] = static_cast<std::uint8_t>(packed & 0xFFU);
        }
        if (hl == 0) { row_scales_g[g] = se; }
    }
}

} // namespace

bool w4_g16_quantizable(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr && w.n > 0 && w.k > 0 &&
           w.k % 16 == 0 && (w.padded_shape[1] == w.k || w.padded_shape[1] <= 1) &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0;
}

std::int32_t w4_g16_padded_columns(std::int32_t k) noexcept { return padded_columns(k); }

Weight quantize_bf16_w4_g16(const Weight& src, void* codes, void* row_scales, void* scales, cudaStream_t stream) {
    if (!w4_g16_quantizable(src) || codes == nullptr || row_scales == nullptr || scales == nullptr ||
        reinterpret_cast<std::uintptr_t>(codes) % 16 != 0 || reinterpret_cast<std::uintptr_t>(row_scales) % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(scales) % 8 != 0) {
        throw std::invalid_argument(
            "quantize_bf16_w4_g16: needs a 16-byte aligned BF16 Contiguous [n, k] weight (k a multiple of 16), "
            "16-byte aligned codes, 4-byte aligned row_scales, and 8-byte aligned scales");
    }
    const int n = src.n, k = src.k, padded_k = padded_columns(k);
    quantize_w4_kernel<<<n, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src.qdata), k, padded_k, static_cast<std::uint8_t*>(codes),
        static_cast<float*>(row_scales), static_cast<std::uint8_t*>(scales));
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        throw std::runtime_error(std::string("quantize_bf16_w4_g16: ") + cudaGetErrorString(e));
    }
    const int groups = padded_k / 16;
    Weight w           = src;
    w.qtype            = QType::NVFP4;
    w.layout           = QuantLayout::RowSplit;
    w.payload          = codes;
    w.payload_bytes    = 0;
    w.high_plane_bytes = static_cast<std::uint64_t>(n) * 4;
    w.qdata            = codes;
    w.qhigh            = row_scales;
    w.scales           = scales;
    w.escapes          = nullptr;
    w.padded_shape[1]  = padded_k;
    w.group_size       = 16;
    w.group            = 16;
    w.scale_dtype      = DType::FP8_E4M3FN;
    w.scale_ne[0]      = groups;
    w.scale_ne[1]      = n;
    w.scale_nb[0]      = 1;
    w.scale_nb[1]      = groups;
    w.scale_nb[2] = w.scale_nb[3] = w.scale_nb[1] * n;
    return w;
}

} // namespace infernix::ops
