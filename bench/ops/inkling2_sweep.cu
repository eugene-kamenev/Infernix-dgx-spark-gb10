// Synthetic GB10 microbenchmark for Inkling 2's fused dense projections (BF16, Q8_G32, W4_G16) across
// projections_dmma (T <= 16), projections (output_head / fallback), and projections_mma (T >= 128).
// Always flushes L2 (48 MiB) between timed kernel invocations and measures kernel-only elapsed microseconds.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/q8_g32.h"
#include "infernix/ops/w4_g16.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace infernix;
namespace ik = infernix::ops::inkling2;

namespace {

__global__ void fill_bf16_pattern(__nv_bfloat16* dst, std::size_t n, std::uint32_t seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    const std::uint32_t h = static_cast<std::uint32_t>(idx) * 2654435761U ^ seed;
    const float v         = (static_cast<float>(static_cast<int>(h & 1023U) - 512) / 512.0F) * 0.25F;
    dst[idx]              = __float2bfloat16_rn(v);
}

void init_bf16(void* dst, std::size_t elems, std::uint32_t seed, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const auto blocks      = static_cast<unsigned>((elems + kThreads - 1) / kThreads);
    fill_bf16_pattern<<<blocks, kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(dst), elems, seed);
}

enum class Fmt { BF16, Q8, W4 };

const char* fmt_name(Fmt f) {
    switch (f) {
    case Fmt::BF16: return "bf16";
    case Fmt::Q8: return "q8";
    case Fmt::W4: return "w4";
    }
    return "?";
}

struct SegSpec {
    std::int32_t rows;
};

struct ShapeSpec {
    const char* name;
    std::int32_t k;
    bool fp32_out;
    bool is_head;
    std::vector<SegSpec> segs;
};

struct BenchSegment {
    std::int32_t rows = 0;
    std::int32_t k    = 0;
    DeviceBuffer bf16;
    DeviceBuffer q8_codes, q8_scales;
    DeviceBuffer w4_codes, w4_row_scales, w4_scales;
    Weight bf16_w{}, q8_w{}, w4_w{};

    void init(std::int32_t r, std::int32_t cols, std::uint32_t seed, cudaStream_t stream) {
        rows             = r;
        k                = cols;
        const auto elems = static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        bf16             = DeviceBuffer((elems * 2 + 255) / 256 * 256);
        init_bf16(bf16.p, elems, seed, stream);

        bf16_w.payload         = bf16.p;
        bf16_w.payload_bytes   = elems * 2;
        bf16_w.qtype           = QType::BF16;
        bf16_w.layout          = QuantLayout::Contiguous;
        bf16_w.ndim            = 2;
        bf16_w.n = bf16_w.shape[0] = bf16_w.padded_shape[0] = r;
        bf16_w.k = bf16_w.shape[1] = bf16_w.padded_shape[1] = cols;
        bf16_w.qdata                                        = bf16.p;

        if (cols % 32 == 0) {
            const std::int32_t pk = ops::q8_g32_padded_columns(cols);
            q8_codes              = DeviceBuffer((static_cast<std::size_t>(r) * pk + 255) / 256 * 256);
            q8_scales             = DeviceBuffer((static_cast<std::size_t>(r) * (pk / 32) * 2 + 255) / 256 * 256);
            q8_w                  = ops::quantize_bf16_q8_g32(bf16_w, q8_codes.p, q8_scales.p, stream);
        }
        if (cols % 16 == 0) {
            const std::int32_t pk = ops::w4_g16_padded_columns(cols);
            w4_codes              = DeviceBuffer((static_cast<std::size_t>(r) * (pk / 2) + 255) / 256 * 256);
            w4_row_scales         = DeviceBuffer((static_cast<std::size_t>(r) * 4 + 255) / 256 * 256);
            w4_scales             = DeviceBuffer((static_cast<std::size_t>(r) * (pk / 16) + 255) / 256 * 256);
            w4_w = ops::quantize_bf16_w4_g16(bf16_w, w4_codes.p, w4_row_scales.p, w4_scales.p, stream);
        }
    }
};

void fill_arg_seg(ik::ProjectionSegment& s, const BenchSegment& bs, Fmt f, void* out) {
    s       = {};
    s.rows  = bs.rows;
    s.input = 0;
    s.out   = out;
    if (f == Fmt::BF16) {
        s.weight = static_cast<const std::uint16_t*>(bs.bf16_w.qdata);
    } else if (f == Fmt::Q8) {
        s.codes     = bs.q8_w.qdata;
        s.q8_scales = static_cast<const std::uint16_t*>(bs.q8_w.scales);
        s.q8_stride = static_cast<std::int32_t>(bs.q8_w.padded_shape[1]);
    } else {
        s.codes         = bs.w4_w.qdata;
        s.w4_scales     = static_cast<const std::uint8_t*>(bs.w4_w.scales);
        s.w4_row_scales = static_cast<const float*>(bs.w4_w.qhigh);
        s.w4_stride     = static_cast<std::int32_t>(bs.w4_w.padded_shape[1]);
    }
}

std::vector<int> parse_tokens(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) { out.push_back(std::stoi(tok)); }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string filter = "all";
    std::string fmt_arg = "all";
    std::vector<int> tokens_list = {1, 2, 4, 8, 16, 128, 1024};
    bool csv                     = false;
    bool flush                   = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        } else if (a == "--fmt" && i + 1 < argc) {
            fmt_arg = argv[++i];
        } else if (a == "--tokens" && i + 1 < argc) {
            tokens_list = parse_tokens(argv[++i]);
        } else if (a == "--csv") {
            csv = true;
        } else if (a == "--no-flush") {
            flush = false;
        }
    }

    DeviceContext dev(0);
    cudaStream_t stream = dev.stream;

    const std::vector<ShapeSpec> shapes = {
        {"k2", 2304, false, false, {{3072}, {192}, {3072}, {24}, {1024}, {1024}}},
        {"w_o", 3072, false, false, {{2304}}},
        {"output_head", 2304, true, true, {{200058}}},
        {"dspark_gu", 2304, false, false, {{9216}, {9216}}},
        {"dspark_down", 9216, false, false, {{2304}}},
    };

    const int max_t = *std::max_element(tokens_list.begin(), tokens_list.end());
    DeviceBuffer x_buf(static_cast<std::size_t>(std::max(max_t, 1)) * 9216 * 2 + 256);
    init_bf16(x_buf.p, static_cast<std::size_t>(std::max(max_t, 1)) * 9216, 0x1234U, stream);
    DeviceBuffer out_buf(static_cast<std::size_t>(std::max(max_t, 1)) * 200058 * 4 + 256);
    DeviceBuffer l2_flush(48ULL << 20);

    cudaEvent_t ev0{}, ev1{};
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    if (csv) {
        std::printf("shape,fmt,route,T,us,gbps,tflops\n");
    } else {
        std::printf("%-12s %-4s %-8s %5s %9s %9s %9s\n", "shape", "fmt", "route", "T", "us", "GB/s", "TFLOP/s");
    }

    std::uint32_t seed = 100;
    for (const ShapeSpec& sp : shapes) {
        if (filter != "all" && filter != sp.name) { continue; }
        std::vector<BenchSegment> bsegs(sp.segs.size());
        std::int64_t total_rows = 0;
        for (std::size_t i = 0; i < sp.segs.size(); ++i) {
            bsegs[i].init(sp.segs[i].rows, sp.k, seed++, stream);
            total_rows += sp.segs[i].rows;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const std::vector<Fmt> fmts =
            fmt_arg == "all"  ? std::vector<Fmt>{Fmt::BF16, Fmt::Q8, Fmt::W4}
            : fmt_arg == "bf16" ? std::vector<Fmt>{Fmt::BF16}
            : fmt_arg == "q8"   ? std::vector<Fmt>{Fmt::Q8}
                                : std::vector<Fmt>{Fmt::W4};

        for (Fmt f : fmts) {
            for (int T : tokens_list) {
                if (sp.is_head && T > 128) { continue; } // Forward::head handles <= 128 cols at a time

                ik::ProjectionArgs args{};
                args.count    = static_cast<std::int32_t>(bsegs.size());
                args.k        = sp.k;
                args.columns  = T;
                args.x[0]     = static_cast<const std::uint16_t*>(x_buf.p);
                args.fp32_out = sp.fp32_out;
                std::size_t off = 0;
                const std::size_t word = sp.fp32_out ? 4 : 2;
                for (std::size_t s = 0; s < bsegs.size(); ++s) {
                    auto* dst = static_cast<std::uint8_t*>(out_buf.p) + off;
                    fill_arg_seg(args.segments[s], bsegs[s], f, dst);
                    off += ((static_cast<std::size_t>(T) * bsegs[s].rows * word + 255) / 256) * 256;
                }

                const bool use_dmma = !sp.is_head && T <= 16;
                const bool use_mma  = !sp.is_head && T > 16;
                const char* route   = use_dmma ? "dmma" : use_mma ? "mma" : "skinny";
                const auto run_once = [&] {
                    if (use_dmma) {
                        ik::projections_dmma(args, stream);
                    } else if (use_mma) {
                        ik::projections_mma(args, stream);
                    } else {
                        ik::projections(args, stream);
                    }
                };

                for (int w = 0; w < 3; ++w) { run_once(); }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const int reps = T <= 16 ? 25 : 11;
                std::vector<float> times;
                times.reserve(reps);
                for (int r = 0; r < reps; ++r) {
                    if (flush) { CUDA_CHECK(cudaMemsetAsync(l2_flush.p, r & 0xFF, l2_flush.bytes, stream)); }
                    CUDA_CHECK(cudaEventRecord(ev0, stream));
                    run_once();
                    CUDA_CHECK(cudaEventRecord(ev1, stream));
                    CUDA_CHECK(cudaEventSynchronize(ev1));
                    float ms = 0.0F;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                    times.push_back(ms * 1000.0F);
                }
                std::sort(times.begin(), times.end());
                const double us = static_cast<double>(times[times.size() / 2]);

                const double elems = static_cast<double>(total_rows) * static_cast<double>(sp.k);
                const std::int32_t pk =
                    f == Fmt::W4 ? ops::w4_g16_padded_columns(sp.k)
                                 : (f == Fmt::Q8 ? ops::q8_g32_padded_columns(sp.k) : sp.k);
                const double w_bytes =
                    f == Fmt::BF16 ? elems * 2.0
                    : f == Fmt::Q8 ? elems + static_cast<double>(total_rows) * (pk / 32) * 2.0
                                   : elems * 0.5 + static_cast<double>(total_rows) * (pk / 16 + 4.0);
                const double io_bytes = w_bytes + static_cast<double>(T) * sp.k * 2.0 +
                                        static_cast<double>(T) * total_rows * static_cast<double>(word);
                const double gbps   = io_bytes / (us * 1e3);
                const double tflops = (2.0 * T * elems) / (us * 1e6);

                if (csv) {
                    std::printf("%s,%s,%s,%d,%.2f,%.2f,%.3f\n", sp.name, fmt_name(f), route, T, us, gbps, tflops);
                } else {
                    std::printf("%-12s %-4s %-8s %5d %9.2f %9.2f %9.3f\n", sp.name, fmt_name(f), route, T, us, gbps,
                                tflops);
                }
            }
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return 0;
}
