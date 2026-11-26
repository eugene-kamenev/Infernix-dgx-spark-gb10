#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"

#include <cuda_fp16.h>

#include <type_traits>

namespace infernix::ops::detail::gated_delta_net::chunked {
namespace {

template <int DV>
inline constexpr int kThreads = DV >= 32 ? 256 : 128;

template <int Width>
__device__ __forceinline__ int value_index(int row, int col) {
    if constexpr (Width == 8) return row * Width + col;
    if constexpr (Width == 16) return row * Width + (col ^ (((row >> 1) & 1) * 8));
    return row * Width + (col ^ ((row & 3) * 8));
}

__device__ __forceinline__ void load_vector_tf32_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    unsigned low, high;
    ldmatrix_x2(low, high, smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    unpack_bf16x2_to_fp32_bits(low, a[0], a[2]);
    unpack_bf16x2_to_fp32_bits(high, a[1], a[3]);
}

// Q/K retain their native BF16 coordinates. TF32 A expands adjacent pairs;
// its FP32 B loads follow the corresponding even/odd inner-coordinate order.
__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    ldmatrix_x4(
        a[0], a[1], a[2], a[3],
        smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 8)));
}

__device__ __forceinline__ void load_transposed_bf16_a(const __nv_bfloat16* p, int key, int lane,
                                                       unsigned (&a)[4]) {
    ldmatrix_x4_t(
        a[0], a[1], a[2], a[3],
        smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8, key + ((lane >> 3) & 1) * 8)));
}

template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                            unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = pack_bf16x2(p[value_index<DV>(k + 2 * t, col)], p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 =
        pack_bf16x2(p[value_index<DV>(k + 2 * t + 8, col)], p[value_index<DV>(k + 2 * t + 9, col)]);
}

// GdnChunkArithmetic::Fp16Accumulate's helpers. Its packet holds FP16 Q/K; the 16-bit ldmatrix
// loads above are type-agnostic.

// The FP32 state and delta operands as FP16, saturating: in range the bits of pack_f16x2 (round to
// nearest even), beyond +-65504 the largest finite value instead of an infinity, which would turn
// the products to NaN and persist in the FP32 state for the rest of the sequence.
__device__ __forceinline__ std::uint32_t pack_f16x2_sat(float lo, float hi) {
    std::uint32_t out;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

// Two already-FP16 elements as one MMA B fragment pair; the snapshot stores them pre-saturated.
__device__ __forceinline__ std::uint32_t pack_half2(__half lo, __half hi) {
    return std::uint32_t(__half_as_ushort(lo)) | (std::uint32_t(__half_as_ushort(hi)) << 16);
}

template <int DV>
__device__ __forceinline__ void load_f16_b(const __half* p, int k, int n, int lane, unsigned& b0,
                                           unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = pack_half2(p[value_index<DV>(k + 2 * t, col)], p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 = pack_half2(p[value_index<DV>(k + 2 * t + 8, col)],
                    p[value_index<DV>(k + 2 * t + 9, col)]);
}

// On GB10 (sm_121a), m16n8k16.row.col.f32.f16.f16.f32 runs at full FP16 Tensor Core throughput
// (125.31 TFLOP/s, 1:1 with f16acc), so Fp16Accumulate accumulates directly in FP32 without
// intermediate FP16 partial-sum rounding or half2-to-float2 unpacks.

template <int DV, bool Fp16>
struct RecurrenceShared {
    struct alignas(256) Stage {
        QkChunk qk;
        ControlChunk control;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];

    // Fp16 operands consume the state snapshot as FP16 (16 KiB at DV=64). When DV/8 >= kWarps
    // (DV=64), each warp owns a disjoint 8-column slice across all 128 keys, so the exact route
    // stages 16 keys per k-step (4 KiB at DV=64), unlocking 2 CTAs/SM on both routes.
    static constexpr int kSnapshotRows =
        (!Fp16 && (DV / 8 >= kThreads<DV> / 32)) ? kChunkSize : kStateDim;
    using Snapshot = std::conditional_t<Fp16, __half, float>;
    Snapshot snapshot[kSnapshotRows * DV];
    float delta[kChunkSize * DV];
};

template <int DV, bool Fp16>
__device__ __forceinline__ void load_stage(typename RecurrenceShared<DV, Fp16>::Stage& stage,
                                           const Arguments& args, const QkChunk* qk_workspace,
                                           const ControlChunk* control_workspace, int head,
                                           int value_start, int chunk, int tid) {
    const int qh       = head / (args.value_heads / args.qk_heads);
    const int chunks   = chunk_count(args.tokens);
    const auto* qk_src = reinterpret_cast<const uint4*>(
        qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
    auto* qk_dst = reinterpret_cast<uint4*>(&stage.qk);
#pragma unroll
    for (int i = tid; i < sizeof(QkChunk) / 16; i += kThreads<DV>)
        cp_async<16, Cache::cg>(qk_dst + i, qk_src + i);
    const auto* ctrl_src = reinterpret_cast<const uint4*>(
        control_workspace + static_cast<std::int64_t>(head) * chunks + chunk);
    auto* ctrl_dst = reinterpret_cast<uint4*>(&stage.control);
#pragma unroll
    for (int i = tid; i < sizeof(ControlChunk) / 16; i += kThreads<DV>)
        cp_async<16, Cache::cg>(ctrl_dst + i, ctrl_src + i);
    constexpr int vectors = kChunkSize * DV / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += kThreads<DV>) {
        const int row = i / (DV / 8), col = i % (DV / 8) * 8;
        const int token       = chunk * kChunkSize + row;
        const int valid_token = token < args.tokens ? token : 0;
        const auto off =
            (static_cast<std::int64_t>(valid_token) * args.value_heads + head) * kStateDim +
            value_start + col;
        cp_async_zfill<16, Cache::cg>(stage.value + row * DV + col, args.v + off,
                                      token < args.tokens ? 16 : 0);
    }
    cp_commit();
}

template <int DV>
__device__ __forceinline__ void store_residual(float* delta, const __nv_bfloat16* value,
                                               const float (&prediction)[4], const float* prefix,
                                               int n, int lane) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float2 v0 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 = __bfloat1622float2(load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    store_vec(delta + value_index<DV>(row, col),
              make_float2(v0.x - prefix[row] * prediction[0], v0.y - prefix[row] * prediction[1]));
    store_vec(delta + value_index<DV>(row + 8, col),
              make_float2(v1.x - prefix[row + 8] * prediction[2],
                          v1.y - prefix[row + 8] * prediction[3]));
}

template <int DV>
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* matrix,
                                             const float* delta, int n, int lane) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < kChunkSize; k += 8) {
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(matrix + square_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                    k + (lane >> 4) * 4)));
        mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3],
                      __float_as_uint(delta[value_index<DV>(k + t, n * 8 + g)]),
                      __float_as_uint(delta[value_index<DV>(k + t + 4, n * 8 + g)]));
    }
}

__device__ __forceinline__ void store_output(const Arguments& args, float (&acc)[4], int chunk,
                                             int head, int value_start, int n, int lane) {
    const int row = lane / 4, col = value_start + n * 8 + (lane & 3) * 2;
    const int t0  = chunk * kChunkSize + row;
    const auto o0 = (static_cast<std::int64_t>(t0) * args.value_heads + head) * kStateDim + col;
    if (t0 < args.tokens) {
        store_vec(args.out + o0, __floats2bfloat162_rn(acc[0] * args.scale, acc[1] * args.scale));
    }
    if (t0 + 8 < args.tokens) {
        store_vec(args.out + o0 + static_cast<std::int64_t>(8) * args.value_heads * kStateDim,
                  __floats2bfloat162_rn(acc[2] * args.scale, acc[3] * args.scale));
    }
}

// fp16-gdn-state: the recurrent kernels' clamp (gated_delta_net/common.cuh), so a stored state
// saturates at FP16's largest finite magnitude instead of overflowing.
__device__ __forceinline__ float saturate_fp16_state(float value) {
    constexpr float kMaximum = 65504.0f;
    return value == value ? fminf(fmaxf(value, -kMaximum), kMaximum) : value;
}

__device__ __forceinline__ float stored_state(float value) { return value; }
__device__ __forceinline__ float stored_state(__half value) { return __half2float(value); }
__device__ __forceinline__ float stored_state(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class State>
__device__ __forceinline__ State state_to_store(float value) {
    if constexpr (std::is_same_v<State, float>) {
        return value;
    } else if constexpr (std::is_same_v<State, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else {
        return __float2half_rn(saturate_fp16_state(value));
    }
}

// Fp16 selects GdnChunkArithmetic::Fp16Accumulate (launch.h); without it the exact arithmetic. State is
// the stored state's element type (float, or __half under fp16-gdn-state).
template <int DV, bool Fp16 = false, class State = float>
__global__ __launch_bounds__(kThreads<DV>, (DV >= 32 ? 2 : 4)) void chunk_recurrence_kernel(
    Arguments args, const QkChunk* __restrict__ qk_workspace,
    const ControlChunk* __restrict__ control_workspace) {
    const State* __restrict__ state_in = static_cast<const State*>(args.state_in);
    State* __restrict__ state_out      = static_cast<State*>(args.state_out);
    constexpr int kWarps = kThreads<DV> / 32;
    constexpr int NT     = DV / 8;
    constexpr int MW     = NT < kWarps ? kWarps / NT : 1;
    constexpr int MPW    = (kStateDim / 16) / MW;
    constexpr int NPW    = NT > kWarps ? NT / kWarps : 1;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm      = *reinterpret_cast<RecurrenceShared<DV, Fp16>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head        = blockIdx.x / (kStateDim / DV);
    const int value_start = blockIdx.x % (kStateDim / DV) * DV;
    const int n_start     = warp % NT;
    const int m_start     = NT < kWarps ? warp / NT : 0;
    const int chunks      = chunk_count(args.tokens);
    const auto state_base = static_cast<std::int64_t>(head) * kStateDim * kStateDim;

    // MMA accumulators describe S^T[key,value]. External state remains [value,key].
    float state[NPW][MPW][4];
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key    = (m_start + mi * MW) * 16 + g;
            const int value  = value_start + n * 8 + t * 2;
            state[ni][mi][0] = stored_state(state_in[state_base + value * kStateDim + key]);
            state[ni][mi][1] = stored_state(state_in[state_base + (value + 1) * kStateDim + key]);
            state[ni][mi][2] = stored_state(state_in[state_base + value * kStateDim + key + 8]);
            state[ni][mi][3] = stored_state(state_in[state_base + (value + 1) * kStateDim + key + 8]);
        }
    }
    load_stage<DV, Fp16>(sm.stage[0], args, qk_workspace, control_workspace, head, value_start, 0,
                         tid);
    if (chunks > 1)
        load_stage<DV, Fp16>(sm.stage[1], args, qk_workspace, control_workspace, head, value_start,
                             1, tid);

    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk + 1 < chunks)
            cp_wait<1>();
        else
            cp_wait<0>();
        if constexpr (Fp16 || NT < kWarps) {
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = n_start + ni * kWarps;
#pragma unroll
                for (int mi = 0; mi < MPW; ++mi) {
                    const int key = (m_start + mi * MW) * 16 + g;
                    const int col = n * 8 + t * 2;
                    if constexpr (Fp16) {
                        // Saturating FP16 rounding done once; every k-step reuses these bits.
                        *reinterpret_cast<std::uint32_t*>(sm.snapshot + value_index<DV>(key, col)) =
                            pack_f16x2_sat(state[ni][mi][0], state[ni][mi][1]);
                        *reinterpret_cast<std::uint32_t*>(sm.snapshot +
                                                          value_index<DV>(key + 8, col)) =
                            pack_f16x2_sat(state[ni][mi][2], state[ni][mi][3]);
                    } else {
                        store_vec(sm.snapshot + value_index<DV>(key, col),
                                  make_float2(state[ni][mi][0], state[ni][mi][1]));
                        store_vec(sm.snapshot + value_index<DV>(key + 8, col),
                                  make_float2(state[ni][mi][2], state[ni][mi][3]));
                    }
                }
            }
        }
        __syncthreads();
        const auto& stage     = sm.stage[chunk & 1];
        const ControlChunk& p = stage.control;
        const QkChunk& qk     = stage.qk;
        float history[NPW][4]{};

        if constexpr (NT >= kWarps) {
            float prediction[NPW][4]{};
            if constexpr (!Fp16) {
#pragma unroll
                for (int mi = 0; mi < MPW; ++mi) {
                    const int k = mi * 16;
                    unsigned ka0[4], ka1[4], qa[4];
                    load_vector_tf32_a(qk.k, k, lane, ka0);
                    load_vector_tf32_a(qk.k, k + 8, lane, ka1);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < NPW; ++ni) {
                        const int n   = warp + ni * kWarps;
                        const int col = n * 8 + t * 2;
                        store_vec(sm.snapshot + value_index<DV>(g, col),
                                  make_float2(state[ni][mi][0], state[ni][mi][1]));
                        store_vec(sm.snapshot + value_index<DV>(g + 8, col),
                                  make_float2(state[ni][mi][2], state[ni][mi][3]));
                        __syncwarp();
                        const float b0 = sm.snapshot[value_index<DV>(2 * t, n * 8 + g)];
                        const float b1 = sm.snapshot[value_index<DV>(2 * t + 1, n * 8 + g)];
                        const float b2 = sm.snapshot[value_index<DV>(2 * t + 8, n * 8 + g)];
                        const float b3 = sm.snapshot[value_index<DV>(2 * t + 9, n * 8 + g)];
                        __syncwarp();
                        mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2],
                                      prediction[ni][3], ka0[0], ka0[1], ka0[2], ka0[3],
                                      __float_as_uint(b0), __float_as_uint(b1));
                        mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2],
                                      prediction[ni][3], ka1[0], ka1[1], ka1[2], ka1[3],
                                      __float_as_uint(b2), __float_as_uint(b3));
                        mma_bf16(history[ni][0], history[ni][1], history[ni][2], history[ni][3],
                                 qa[0], qa[1], qa[2], qa[3], pack_bf16x2(b0, b1),
                                 pack_bf16x2(b2, b3));
                    }
                }
            } else {
                // S was rounded to FP16 (saturating) once per element when the snapshot was
                // stored; both products consume those bits, accumulated in FP32.
#pragma unroll
                for (int k = 0; k < kStateDim; k += 16) {
                    unsigned ka[4], qa[4];
                    load_vector_bf16_a(qk.k, k, lane, ka);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < NPW; ++ni) {
                        const int n = warp + ni * kWarps;
                        const unsigned s0 =
                            pack_half2(sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)],
                                       sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)]);
                        const unsigned s1 =
                            pack_half2(sm.snapshot[value_index<DV>(k + 2 * t + 8, n * 8 + g)],
                                       sm.snapshot[value_index<DV>(k + 2 * t + 9, n * 8 + g)]);
                        mma_f16(prediction[ni][0], prediction[ni][1], prediction[ni][2],
                                prediction[ni][3], ka[0], ka[1], ka[2], ka[3], s0, s1);
                        mma_f16(history[ni][0], history[ni][1], history[ni][2], history[ni][3],
                                qa[0], qa[1], qa[2], qa[3], s0, s1);
                    }
                }
            }
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                store_residual<DV>(sm.delta, stage.value, prediction[ni], p.prefix,
                                   warp + ni * kWarps, lane);
            }
        } else {
            if (warp < 2 * NT) {
                const bool query = warp >= NT;
                const int n      = warp % NT;
                if (query) {
                    if constexpr (!Fp16) {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.q, k, lane, af);
                            load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_bf16(history[0][0], history[0][1], history[0][2], history[0][3],
                                     af[0], af[1], af[2], af[3], b0, b1);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.q, k, lane, af);
                            load_f16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_f16(history[0][0], history[0][1], history[0][2], history[0][3],
                                    af[0], af[1], af[2], af[3], b0, b1);
                        }
                    }
                } else {
                    if constexpr (Fp16) {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.k, k, lane, af);
                            load_f16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            mma_f16(history[0][0], history[0][1], history[0][2], history[0][3],
                                    af[0], af[1], af[2], af[3], b0, b1);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 8) {
                            unsigned af[4];
                            load_vector_tf32_a(qk.k, k, lane, af);
                            mma_tf32_bits(
                                history[0][0], history[0][1], history[0][2], history[0][3], af[0],
                                af[1], af[2], af[3],
                                __float_as_uint(sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)]),
                                __float_as_uint(
                                    sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)]));
                        }
                    }
                    store_residual<DV>(sm.delta, stage.value, history[0], p.prefix, n, lane);
                }
            }
        }
        __syncthreads();

        // Each value tile reads only its own R columns, so R can become Delta in place.
#pragma unroll
        for (int n = warp; n < NT; n += kWarps) {
            float delta[4]{};
            matrix_delta<DV>(delta, p.solve, sm.delta, n, lane);
            const int col = n * 8 + t * 2;
            __syncwarp();
            store_vec(sm.delta + value_index<DV>(g, col), make_float2(delta[0], delta[1]));
            store_vec(sm.delta + value_index<DV>(g + 8, col), make_float2(delta[2], delta[3]));
        }
        __syncthreads();

        if constexpr (NT >= kWarps) {
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = warp + ni * kWarps;
                history[ni][0] *= p.prefix[g];
                history[ni][1] *= p.prefix[g];
                history[ni][2] *= p.prefix[g + 8];
                history[ni][3] *= p.prefix[g + 8];
                matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane);
                store_output(args, history[ni], chunk, head, value_start, n, lane);
            }
        } else if (warp >= NT && warp < 2 * NT) {
            const int n = warp - NT;
            history[0][0] *= p.prefix[g];
            history[0][1] *= p.prefix[g];
            history[0][2] *= p.prefix[g + 8];
            history[0][3] *= p.prefix[g + 8];
            matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane);
            store_output(args, history[0], chunk, head, value_start, n, lane);
        }

        if constexpr (!Fp16) {
            // S^T = gamma S^T + K^T diag(suffix) Delta. The master state stays FP32.
            // Convert Delta only for this MMA and reuse its fragment across all key tiles.
            // The output branch above consumes the original FP32 Delta through TF32 MMA.
            unsigned delta_b[NPW][2];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int col = (n_start + ni * kWarps) * 8 + g;
                delta_b[ni][0] =
                    pack_bf16x2(p.suffix[2 * t] * sm.delta[value_index<DV>(2 * t, col)],
                                p.suffix[2 * t + 1] * sm.delta[value_index<DV>(2 * t + 1, col)]);
                delta_b[ni][1] =
                    pack_bf16x2(p.suffix[2 * t + 8] * sm.delta[value_index<DV>(2 * t + 8, col)],
                                p.suffix[2 * t + 9] * sm.delta[value_index<DV>(2 * t + 9, col)]);
            }
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key     = (m_start + mi * MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    state[ni][mi][0] *= decay;
                    state[ni][mi][1] *= decay;
                    state[ni][mi][2] *= decay;
                    state[ni][mi][3] *= decay;
                }
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    mma_bf16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2],
                             state[ni][mi][3], af[0], af[1], af[2], af[3], delta_b[ni][0],
                             delta_b[ni][1]);
                }
            }
        } else {
            // The same update with FP16 operands. Each tile's 16-token product is rounded to FP16
            // once, from zero, and carried into the FP32 state together with the decay.
            unsigned delta_b[NPW][2];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int col = (n_start + ni * kWarps) * 8 + g;
                delta_b[ni][0] =
                    pack_f16x2_sat(p.suffix[2 * t] * sm.delta[value_index<DV>(2 * t, col)],
                               p.suffix[2 * t + 1] * sm.delta[value_index<DV>(2 * t + 1, col)]);
                delta_b[ni][1] =
                    pack_f16x2_sat(p.suffix[2 * t + 8] * sm.delta[value_index<DV>(2 * t + 8, col)],
                               p.suffix[2 * t + 9] * sm.delta[value_index<DV>(2 * t + 9, col)]);
            }
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key     = (m_start + mi * MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    state[ni][mi][0] *= decay;
                    state[ni][mi][1] *= decay;
                    state[ni][mi][2] *= decay;
                    state[ni][mi][3] *= decay;
                    mma_f16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2], state[ni][mi][3],
                            af[0], af[1], af[2], af[3], delta_b[ni][0], delta_b[ni][1]);
                }
            }
        }
        __syncthreads();
        if (chunk + 2 < chunks) {
            load_stage<DV, Fp16>(sm.stage[chunk & 1], args, qk_workspace, control_workspace, head,
                                 value_start, chunk + 2, tid);
        }
    }
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key                                        = (m_start + mi * MW) * 16 + g;
            const int value                                      = value_start + n * 8 + t * 2;
            state_out[state_base + value * kStateDim + key]           = state_to_store<State>(state[ni][mi][0]);
            state_out[state_base + (value + 1) * kStateDim + key]     = state_to_store<State>(state[ni][mi][1]);
            state_out[state_base + value * kStateDim + key + 8]       = state_to_store<State>(state[ni][mi][2]);
            state_out[state_base + (value + 1) * kStateDim + key + 8] = state_to_store<State>(state[ni][mi][3]);
        }
    }
}

template <int DV, bool Fp16, class State>
void launch(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
            cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV, Fp16>);
    static bool configured = false;
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(chunk_recurrence_kernel<DV, Fp16, State>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        configured = true;
    }
    chunk_recurrence_kernel<DV, Fp16, State>
        <<<args.value_heads*(kStateDim / DV), kThreads<DV>, shared_bytes, stream>>>(args, qk,
                                                                                    control);
    CUDA_CHECK(cudaGetLastError());
}

template <int DV, class State>
void dispatch_state(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
                    cudaStream_t stream, bool fp16) {
    if (fp16) return launch<DV, true, State>(args, qk, control, stream);
    return launch<DV, false, State>(args, qk, control, stream);
}

template <int DV>
void dispatch(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
              cudaStream_t stream, bool fp16) {
    if (args.state_dtype == DType::BF16) {
        return dispatch_state<DV, __nv_bfloat16>(args, qk, control, stream, fp16);
    }
    if (args.state_dtype == DType::FP16) {
        return dispatch_state<DV, __half>(args, qk, control, stream, fp16);
    }
    return dispatch_state<DV, float>(args, qk, control, stream, fp16);
}

} // namespace

int value_tile(int value_heads, int multiprocessor_count) {
    // Small grids need the narrowest stripe. Otherwise the launch is value_heads * (kStateDim / tile)
    // CTAs and the widest stripe that still fills two SM waves wins: wider stripes quarter the
    // per-chunk Qk/Control packet re-reads each slice pays (measured on GB10 sm_121a, 48 SMs,
    // Hv=48: tile 64 is 13-19% faster than tile 16 at T=512..8192; tile 32 fills wave 2 at Hv=32
    // where tile 64 underfills it).
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    for (const int tile : {64, 32}) {
        if (static_cast<std::int64_t>(value_heads) * (kStateDim / tile) >= 2 * multiprocessor_count) return tile;
    }
    return 16;
}

namespace {

int select_recurrence_value_tile(int qk_heads, int value_heads, int tokens,
                                 int multiprocessor_count) {
    if (qk_heads == 8 && value_heads == 24) return tokens <= 128 ? 32 : 64;
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    if (tokens <= 128) return 16;
    return value_tile(value_heads, multiprocessor_count);
}

} // namespace

void launch_recurrence(const Arguments& args, const QkChunk* qk, const ControlChunk* control,
                       DeviceExecutionView execution, bool fp16) {
    switch (select_recurrence_value_tile(args.qk_heads, args.value_heads, args.tokens,
                                         execution.multiprocessor_count)) {
    case 64:
        return dispatch<64>(args, qk, control, execution.stream, fp16);
    case 32:
        return dispatch<32>(args, qk, control, execution.stream, fp16);
    case 16:
        return dispatch<16>(args, qk, control, execution.stream, fp16);
    default:
        return dispatch<8>(args, qk, control, execution.stream, fp16);
    }
}

} // namespace infernix::ops::detail::gated_delta_net::chunked
