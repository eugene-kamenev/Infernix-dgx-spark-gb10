// Qwen3.8-Flash-Next fast-path selection (models/qwen4_exp/program/fast_paths.h): where each fast
// path's kernels run, which configurations the registry qualifies it for, how a configuration is
// described, and the routes --a8-activations gives it. The selection itself (select_fast_paths) needs loaded parameters; the startup log and
// the quality gate's runs cover it.

#include "models/qwen4_exp/program/fast_paths.h"

#include <iostream>
#include <string>

namespace {

using infernix::FastPath;
using infernix::KvCacheStorage;
using infernix::QType;
using infernix::models::qwen4_exp::DenseKernels;
using infernix::models::qwen4_exp::FastPathConfiguration;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    namespace q = infernix::models::qwen4_exp;
    int failures = 0;

    const infernix::EngineOptions options;
    const FastPathConfiguration dense8{DenseKernels::Q8, true, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::Q8SlicedK, dense8, options), "q8-sliced-k runs on Q8 dense weights");
    for (const DenseKernels dense : {DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Ggml}) {
        failures += check(!q::fast_path_applies(FastPath::Q8SlicedK, {dense, true, KvCacheStorage::Int8Group64}, options),
                          "q8-sliced-k has no kernels for other dense formats");
    }
    // The Qwen3.8-27B attention paths have no Flash-Next kernels.
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, dense8, options), "prompt-pv8 does not apply");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, dense8, options), "decode-split64 does not apply");

    failures += check(q::describe(dense8) == "Q8 dense, 4-bit expert activations, int8 KV",
                      "configuration description changed");
    failures += check(q::describe({DenseKernels::Fp8Row, false, KvCacheStorage::BFloat16}) ==
                          "FP8 dense, 16-bit expert activations, bf16 KV",
                      "configuration description changed");
    // Every GGUF block format is one dense kernel family, so every GGUF import shares a configuration.
    for (const QType qtype : {QType::GGML_Q4_K, QType::GGML_Q6_K, QType::GGML_IQ3_S, QType::GGML_IQ4_XS}) {
        failures += check(q::dense_kernels(qtype) == DenseKernels::Ggml, "a GGUF block format is not the GGUF family");
    }
    failures += check(q::dense_kernels(QType::Q8_G32_FP16) == DenseKernels::Q8 &&
                          q::dense_kernels(QType::FP8_E4M3FN_ROW_BF16) == DenseKernels::Fp8Row &&
                          q::dense_kernels(QType::BF16) == DenseKernels::Bf16,
                      "a dense format maps to the wrong kernel family");
    failures += check(q::describe({DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}) ==
                          "GGUF dense, 16-bit expert activations, int8 KV",
                      "GGUF configuration description changed");

    failures += check(q::fast_path_applies(FastPath::GdnConvFused, dense8, options), "gdn-conv-fused runs on Q8 dense weights");
    failures += check(!q::fast_path_applies(FastPath::GdnConvFused, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}, options),
                      "gdn-conv-fused has no kernels for BF16 dense weights");
    failures += check(q::fast_path_applies(FastPath::QsaPrepareFused, dense8, options), "qsa-prepare-fused runs on INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Vq2}, options),
                      "qsa-prepare-fused has no VQ2 kernel");

    // The kernel paths are qualified with Q8 dense weights and 4-bit experts on every KV format their
    // kernels serve: qsa-prepare-fused has none for VQ2 or K4V2.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration config{DenseKernels::Q8, true, kv};
        failures += check(q::fast_path_qualified(FastPath::Q8SlicedK, config) &&
                              q::fast_path_qualified(FastPath::GdnConvFused, config),
                          "the dense kernel paths are qualified on every KV format");
        failures += check(q::fast_path_qualified(FastPath::QsaPrepareFused, config) ==
                              q::fast_path_applies(FastPath::QsaPrepareFused, config, options),
                          "qsa-prepare-fused is qualified exactly where its kernel runs");
    }
    for (const FastPath path : {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused}) {
        failures += check(!q::fast_path_qualified(path, {DenseKernels::Q8, false, KvCacheStorage::Int8Group64}),
                          "a kernel path stays unqualified with 16-bit expert activations");
    }
    // fp8-a8-pair runs on recipe C (FP8 dense, 16-bit expert activations) where its own decode gate
    // passed; INT8, which fails its tail, by the owner's decision (2026-10-24).
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value})
        failures += check(q::fast_path_qualified(FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, kv}),
                          "fp8-a8-pair is registered on recipe C with every KV format");
    failures += check(!q::fast_path_qualified(FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, true, KvCacheStorage::Int8Group64}),
                      "fp8-a8-pair is not qualified with 4-bit expert activations");

    // The request-coupling paths are qualified with Q8 dense weights and 4-bit experts on every KV
    // format: speculation and cache resume by their joint arm, preemption by its increment arm.
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration config{DenseKernels::Q8, true, kv};
        for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
            failures += check(q::fast_path_qualified(path, config),
                              "the request-coupling paths are qualified on every KV format");
        }
    }

    q::FastPathSelection selection{.configuration = dense8, .paths = {}};
    const auto exact = selection.forward();
    failures += check(exact.dense_routes == infernix::ops::LinearRoutes::Exact && !exact.gdn_conv_fused &&
                          !exact.qsa_prepare_fused,
                      "the exact route without a selected path");
    selection.paths = {FastPath::Q8SlicedK, FastPath::GdnConvFused, FastPath::QsaPrepareFused};
    const auto fast = selection.forward();
    failures += check(fast.dense_routes == infernix::ops::LinearRoutes::Q8SlicedK && fast.gdn_conv_fused &&
                          fast.qsa_prepare_fused,
                      "each selected path sets its Forward flag");
    // q8-sliced-k's reduction order depends on the round's width; the fused kernels are per column.
    failures += check(!selection.request_invariant(), "q8-sliced-k couples a request to its round");
    selection.paths = {FastPath::GdnConvFused, FastPath::QsaPrepareFused};
    failures += check(selection.request_invariant(), "the fused kernels keep a request invariant");

    // expert-a4-residual acts only on W4A16 experts (recipe C, NVIDIA FP4-Local and W4Dense with
    // --a16-experts), and routes an expert by its column count in the call, which batched rounds share
    // between requests.
    const FastPathConfiguration recipe_c{DenseKernels::Fp8Row, false, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::ExpertA4Residual, recipe_c, options),
                      "expert-a4-residual runs on W4A16 experts");
    failures += check(q::fast_path_applies(FastPath::ExpertA4Residual,
                                           {DenseKernels::Nvfp4, false, KvCacheStorage::Int8Group64}, options),
                      "expert-a4-residual runs on W4Dense's W4A16 experts (task T8, 2026-11-07)");
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, dense8, options),
                      "expert-a4-residual does not apply to W4A4 experts");
    for (const DenseKernels dense : {DenseKernels::Ggml, DenseKernels::Bf16}) {
        failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual, {dense, false, KvCacheStorage::Int8Group64}, options),
                          "expert-a4-residual has no kernel for GGUF or block-FP8 experts");
    }
    failures += check(exact.expert_a4_terms == 0, "the exact route keeps W4A16 experts exact");
    selection.paths = {FastPath::ExpertA4Residual};
    failures += check(selection.forward().expert_a4_terms == 2, "expert-a4-residual takes two terms");
    failures += check(!selection.request_invariant(), "expert-a4-residual couples a request to its round");
    // expert-a4-residual-3, its more precise form, applies where it does and takes precedence.
    for (const auto& configuration : {recipe_c, dense8, FastPathConfiguration{DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64},
                                      FastPathConfiguration{DenseKernels::Mxfp8, true, KvCacheStorage::Int8Group64}}) {
        failures += check(q::fast_path_applies(FastPath::ExpertA4Residual3, configuration, options) ==
                              q::fast_path_applies(FastPath::ExpertA4Residual, configuration, options),
                          "expert-a4-residual-3 applies exactly where expert-a4-residual does");
    }
    // W4Dense registers the two-term pair, not the three-term form (task T8, 2026-11-07).
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual3,
                                            {DenseKernels::Nvfp4, false, KvCacheStorage::Int8Group64}, options),
                      "expert-a4-residual-3 stays off W4Dense");
    selection.paths = {FastPath::ExpertA4Residual, FastPath::ExpertA4Residual3};
    failures += check(selection.forward().expert_a4_terms == 3, "expert-a4-residual-3 takes three terms, first");
    failures += check(!selection.request_invariant(), "expert-a4-residual-3 couples a request to its round");
    // fp8-a8-pair admits a dense call by its width, which batched rounds share.
    selection.paths = {FastPath::Fp8A8Pair};
    failures += check(!selection.request_invariant(), "fp8-a8-pair couples a request to its round");
    // fp8-a8-pair-block32 runs where fp8-a8-pair does and also on FP4-Local's MXFP8 dense (whose group
    // scales feed the same block-scaled MMA), is qualified wherever fp8-a8-pair is (row FP8 only; the owner's
    // decision of 2026-10-25: tied in speed and more precise), couples requests the same way, and sets its own
    // route flag (the linear dispatch gives it precedence when both are selected).
    const FastPathConfiguration mxfp8{DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64};
    for (const auto& configuration : {recipe_c, dense8, FastPathConfiguration{DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}}) {
        failures += check(q::fast_path_applies(FastPath::Fp8A8PairBlock32, configuration, options) ==
                              q::fast_path_applies(FastPath::Fp8A8Pair, configuration, options),
                          "fp8-a8-pair-block32 applies exactly where fp8-a8-pair does");
    }
    failures += check(q::fast_path_applies(FastPath::Fp8A8PairBlock32, mxfp8, options) &&
                          !q::fast_path_applies(FastPath::Fp8A8Pair, mxfp8, options),
                      "fp8-a8-pair-block32 does not serve MXFP8 dense weights");
    for (const KvCacheStorage kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                    KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                    KvCacheStorage::Q4KeyVq2Value}) {
        for (const DenseKernels dense : {DenseKernels::Fp8Row, DenseKernels::Mxfp8}) {
            failures += check(q::fast_path_qualified(FastPath::Fp8A8PairBlock32, {dense, false, kv}) ==
                                  q::fast_path_qualified(FastPath::Fp8A8Pair, {dense, false, kv}),
                              "fp8-a8-pair-block32 is qualified exactly where fp8-a8-pair is");
        }
    }
    selection.paths = {FastPath::Fp8A8PairBlock32};
    failures += check(!selection.request_invariant(), "fp8-a8-pair-block32 couples a request to its round");
    failures += check(selection.forward().dense_routes == infernix::ops::LinearRoutes::Fp8A8PairBlock32,
                      "fp8-a8-pair-block32 sets its own dense route flag");
    selection.paths = {FastPath::Fp8A8Pair, FastPath::Fp8A8PairBlock32};
    failures += check(selection.forward().dense_routes ==
                          (infernix::ops::LinearRoutes::Fp8A8Pair | infernix::ops::LinearRoutes::Fp8A8PairBlock32),
                      "both pair paths set both flags");
    selection.paths = {FastPath::ExpertA4Residual};
    // Every KV format, BF16 by its precise-reference gates on FP4-Local (the same expert kernels as recipe C).
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, kv}),
                          "expert-a4-residual is qualified on recipe C with every KV format");
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, {DenseKernels::Mxfp8, false, kv}),
                          "expert-a4-residual is qualified on NVIDIA FP4-Local (16-bit experts) with every KV format");
        failures += check(q::fast_path_qualified(FastPath::ExpertA4Residual, {DenseKernels::Nvfp4, false, kv}),
                          "expert-a4-residual is qualified on W4Dense (--a16-experts) with every KV format");
        failures += check(q::fast_path_qualified(FastPath::ExpertA4ResidualDecode, {DenseKernels::Nvfp4, false, kv}),
                          "expert-a4-residual-decode is qualified on W4Dense (--a16-experts) with every KV format");
    }
    failures += check(!q::fast_path_applies(FastPath::ExpertA4Residual,
                                            {DenseKernels::Mxfp8, true, KvCacheStorage::Int8Group64}, options),
                      "expert-a4-residual does not apply to FP4-Local's 4-bit expert activations");
    failures += check(!q::fast_path_qualified(FastPath::ExpertA4Residual,
                                              {DenseKernels::Q8, false, KvCacheStorage::Int8Group64}),
                      "expert-a4-residual is qualified only where its kernels were gated and timed");
    // expert-a4-residual-decode, the narrow form (GPU narrow kernels and the CPU engine), serves the same
    // banks, sets its own flag and couples a request to its round (an expert's route follows its column
    // count). It stays a candidate except on W4Dense, which registers it (task T8, 2026-11-07; its
    // W4Dense checks are above).
    for (const auto& configuration : {recipe_c, dense8, FastPathConfiguration{DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64},
                                      FastPathConfiguration{DenseKernels::Mxfp8, true, KvCacheStorage::Int8Group64},
                                      FastPathConfiguration{DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}}) {
        failures += check(q::fast_path_applies(FastPath::ExpertA4ResidualDecode, configuration, options) ==
                              q::fast_path_applies(FastPath::ExpertA4Residual, configuration, options),
                          "expert-a4-residual-decode applies exactly where expert-a4-residual does");
    }
    failures += check(!exact.expert_a4_narrow, "the exact route keeps W4A16 experts' narrow calls exact");
    selection.paths = {FastPath::ExpertA4ResidualDecode};
    failures += check(selection.forward().expert_a4_narrow && selection.forward().expert_a4_terms == 0,
                      "expert-a4-residual-decode sets the narrow flag alone");
    failures += check(!selection.request_invariant(), "expert-a4-residual-decode couples a request to its round");
    selection.paths = {FastPath::ExpertA4Residual, FastPath::ExpertA4ResidualDecode};
    failures += check(selection.forward().expert_a4_narrow && selection.forward().expert_a4_terms == 2,
                      "expert-a4-residual-decode combines with expert-a4-residual");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        for (const DenseKernels dense : {DenseKernels::Fp8Row, DenseKernels::Mxfp8}) {
            failures += check(!q::fast_path_qualified(FastPath::ExpertA4ResidualDecode, {dense, false, kv}),
                              "expert-a4-residual-decode is an unqualified candidate (--fast-path only)");
        }
    }

    // sampled-mtp-drafts acts on the MTP drafter only; gdn-f16-accumulate on every artifact's GDN
    // prefill, per call, so it keeps a request invariant.
    infernix::EngineOptions mtp = options;
    mtp.speculative.backend     = infernix::SpeculativeBackend::Mtp;
    failures += check(q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, mtp) &&
                          !q::fast_path_applies(FastPath::SampledMtpDrafts, dense8, options),
                      "sampled-mtp-drafts applies with the MTP drafter only");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, kv}),
                          "sampled-mtp-drafts is qualified on Dense8 with every KV format");
    }
    // The other Flash-Next configurations: IQ3_S (GGUF dense, 16-bit experts), the official FP8 (BF16
    // dense, 16-bit experts), recipe A (BF16 dense, 4-bit experts) and recipe C (FP8 dense).
    const DenseKernels kGgml = DenseKernels::Ggml;
    const DenseKernels kBf16 = DenseKernels::Bf16;
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        const FastPathConfiguration iq3s{kGgml, false, kv};
        const FastPathConfiguration fp8{kBf16, false, kv};
        const FastPathConfiguration recipe_a{kBf16, true, kv};
        const FastPathConfiguration nvfp4{DenseKernels::Nvfp4, true, kv};
        // Sampled drafts change no logit: every configuration whose drafter they were timed on or by.
        for (const FastPathConfiguration& config :
             {iq3s, fp8, recipe_a, nvfp4, FastPathConfiguration{DenseKernels::Fp8Row, false, kv}}) {
            failures += check(q::fast_path_qualified(FastPath::SampledMtpDrafts, config),
                              "sampled-mtp-drafts is qualified on every Flash-Next configuration");
        }
        for (const FastPathConfiguration& config : {iq3s, fp8, recipe_a, nvfp4, FastPathConfiguration{DenseKernels::Fp8Row, false, kv}}) {
            for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
                failures += check(q::fast_path_qualified(path, config),
                                  "the coupling paths are qualified on IQ3_S, FP8, recipe A, NVFP4 and recipe C");
            }
            failures += check(q::fast_path_qualified(FastPath::QsaPrepareFused, config) ==
                                  q::fast_path_applies(FastPath::QsaPrepareFused, config, options),
                              "qsa-prepare-fused is qualified where its kernel runs");
        }
        failures += check(q::fast_path_qualified(FastPath::FreeCacheResume, {DenseKernels::Nvfp4, false, kv}),
                          "free-cache-resume is qualified on NVFP4 dense with 16-bit expert activations");
        // NVIDIA FP4-Local, both expert activation precisions: speculation and cache resume by its joint
        // gate, preemption by its increment.
        for (const bool a4 : {false, true}) {
            for (const FastPath path : {FastPath::AdaptiveSpeculation, FastPath::FreeCacheResume, FastPath::Preemption}) {
                failures += check(q::fast_path_qualified(path, {DenseKernels::Mxfp8, a4, kv}),
                                  "the coupling paths are qualified on NVIDIA FP4-Local");
            }
            const FastPathConfiguration mxfp8{DenseKernels::Mxfp8, a4, kv};
            for (const FastPath path : {FastPath::Nvfp4A4Pair, FastPath::Q8SlicedK}) {
                failures += check(q::fast_path_applies(path, mxfp8, options) &&
                                      q::fast_path_qualified(path, mxfp8),
                                  "mixed MXFP8/NVFP4/Q8 paths apply and are qualified on NVIDIA FP4-Local");
            }
        }
        failures += check(q::fast_path_applies(FastPath::Nvfp4A4Pair, nvfp4, options) &&
                              q::fast_path_qualified(FastPath::Nvfp4A4Pair, nvfp4),
                          "nvfp4-a4-pair applies and is qualified on NVFP4 dense");
        failures += check(q::fast_path_qualified(FastPath::GdnF16Accumulate, iq3s) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, fp8) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, recipe_a) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, nvfp4) &&
                              q::fast_path_qualified(FastPath::GdnF16Accumulate, {DenseKernels::Q8, true, kv}),
                          "gdn-f16-accumulate is registered across Flash-Next configurations");
    }
    failures += check(exact.gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Exact,
                      "the exact route keeps the GDN prefill exact");
    selection.paths = {FastPath::GdnF16Accumulate};
    failures += check(q::fast_path_applies(FastPath::GdnF16Accumulate, recipe_c, options) &&
                          selection.forward().gdn_arithmetic == infernix::ops::GdnChunkArithmetic::Fp16Accumulate,
                      "gdn-f16-accumulate applies to every artifact and sets its Forward flag");
    failures += check(selection.request_invariant(), "gdn-f16-accumulate keeps a request invariant");

    // ggml-bf16-prefill acts on GGUF dense projections only, sets the BF16 route flag, and couples a
    // prompt's arithmetic to where its calls start.
    failures += check(q::fast_path_applies(FastPath::GgmlBf16Prefill, {kGgml, false, KvCacheStorage::Int8Group64}, options),
                      "ggml-bf16-prefill runs on GGUF dense weights");
    for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row}) {
        failures += check(!q::fast_path_applies(FastPath::GgmlBf16Prefill, {dense, false, KvCacheStorage::Int8Group64}, options),
                          "ggml-bf16-prefill has no route for non-GGUF dense weights");
    }
    failures += check(!infernix::ops::has_route(exact.prefill_routes, infernix::ops::LinearRoutes::GgmlBf16Wide),
                      "the exact route keeps GGUF projections exact");
    selection.paths = {FastPath::GgmlBf16Prefill};
    failures += check(selection.forward().prefill_routes == infernix::ops::LinearRoutes::GgmlBf16Wide &&
                          selection.forward().dense_routes == infernix::ops::LinearRoutes::Exact,
                      "ggml-bf16-prefill sets its route flag for prefill calls only");
    failures += check(!selection.request_invariant(), "ggml-bf16-prefill couples a prompt to its call boundaries");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::GgmlBf16Prefill, {kGgml, false, kv}),
                          "ggml-bf16-prefill is qualified on GGUF imports with every KV format");
    }

    // Two-row rounds are adaptive-speculation's with the MTP drafter and dflash2-pairs' with DFlash2,
    // each only with its drafter, so MTP's qualifications never select DFlash2 pairs. dflash2-pairs is
    // registered on every configuration, and it couples the rows of its rounds.
    infernix::EngineOptions dflash2 = options;
    dflash2.speculative.backend     = infernix::SpeculativeBackend::DFlash2;
    failures += check(q::fast_path_applies(FastPath::DFlash2Pairs, dense8, dflash2) &&
                          !q::fast_path_applies(FastPath::DFlash2Pairs, dense8, mtp) &&
                          !q::fast_path_applies(FastPath::DFlash2Pairs, dense8, options),
                      "dflash2-pairs applies with the DFlash2 drafter only");
    failures += check(!q::fast_path_applies(FastPath::AdaptiveSpeculation, dense8, dflash2),
                      "adaptive-speculation (MTP pairs) does not apply with the DFlash2 drafter");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Ggml,
                                         DenseKernels::Mxfp8, DenseKernels::Nvfp4}) {
            // Every Flash-Next artifact's configuration (the owner's decision, 2026-10-24; W4Dense's
            // NVFP4 dense kind joins it on 2026-10-30): Q8 with 4-bit experts (Dense8), GGUF and FP8
            // with 16-bit (IQ3_S, recipe C), BF16, MXFP8 and NVFP4 with both.
            for (const bool a4 : {false, true}) {
                if (dense == DenseKernels::Fp8Row && a4) { continue; }
                if (dense == DenseKernels::Q8 && !a4) { continue; }
                if (dense == DenseKernels::Ggml && a4) { continue; }
                failures += check(q::fast_path_qualified(FastPath::DFlash2Pairs, {dense, a4, kv}),
                                  "dflash2-pairs is registered on " + q::describe({dense, a4, kv}));
            }
        }
    }
    selection.paths = {FastPath::DFlash2Pairs};
    failures += check(!selection.request_invariant(), "dflash2-pairs couples the rows of a two-row round");
    failures += check(infernix::parse_fast_path("dflash2-pairs") == FastPath::DFlash2Pairs, "dflash2-pairs parses");

    // W4Dense (NVFP4 dense, both expert activation precisions, every KV format): q8-sliced-k routes
    // the artifact's Q8 weights (its NVFP4 problems ignore the flag) and dflash2-pairs joins the
    // owner's every-configuration registration; both are registered. gdn-conv-fused's [16384,2560]
    // NVFP4 route is registered on both expert precisions by the owner's commit 98aff48b0
    // (2026-11-05); its GB10 measurements through 2026-11-01 had it at an Op-level tie and an E2E
    // -0.2 .. +0.03 tok/s against the composed decode route.
    // The fused
    // convolution is per column and keeps a request invariant;
    // q8-sliced-k and dflash2-pairs couple a request to its round, which the reproducible route
    // therefore excludes.
    for (const bool a4 : {false, true}) {
        for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                              KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                              KvCacheStorage::Q4KeyVq2Value}) {
            const FastPathConfiguration w4dense{DenseKernels::Nvfp4, a4, kv};
            failures += check(q::fast_path_applies(FastPath::GdnConvFused, w4dense, options) &&
                                  q::fast_path_qualified(FastPath::GdnConvFused, w4dense),
                              "gdn-conv-fused's NVFP4 route is registered by the owner (98aff48b0)");
            failures += check(q::fast_path_applies(FastPath::Q8SlicedK, w4dense, options) &&
                                  q::fast_path_qualified(FastPath::Q8SlicedK, w4dense),
                              "q8-sliced-k applies and is registered on W4Dense (its Q8 weights)");
            failures += check(q::fast_path_applies(FastPath::RouterMma, w4dense, options) &&
                                  q::fast_path_qualified(FastPath::RouterMma, w4dense),
                              "router-mma applies and is registered on W4Dense");
            if (a4) {
                failures += check(q::fast_path_applies(FastPath::ExpertA4OmmaNarrow, w4dense, options) &&
                                      q::fast_path_qualified(FastPath::ExpertA4OmmaNarrow, w4dense) &&
                                      infernix::parse_fast_path("expert-a4-omma-narrow") ==
                                          FastPath::ExpertA4OmmaNarrow,
                                  "expert-a4-omma-narrow applies and is registered on W4Dense (A4 only)");
                const q::FastPathSelection omma_sel{
                    .configuration = w4dense, .paths = {FastPath::ExpertA4OmmaNarrow}};
                failures += check(omma_sel.forward().expert_a4_omma_narrow && !omma_sel.request_invariant(),
                                  "expert-a4-omma-narrow sets forward flag and stays outside reproducible");
            } else {
                failures += check(!q::fast_path_applies(FastPath::ExpertA4OmmaNarrow, w4dense, options) &&
                                      !q::fast_path_qualified(FastPath::ExpertA4OmmaNarrow, w4dense),
                                  "expert-a4-omma-narrow stays off W4Dense with A16 experts");
            }
        }
    }
    {
        const FastPathConfiguration w4dense_fp8{DenseKernels::Nvfp4, true, KvCacheStorage::Fp8E4M3Row256};
        failures += check(q::fast_path_applies(FastPath::DFlash2Pairs, w4dense_fp8, dflash2) &&
                              q::fast_path_qualified(FastPath::DFlash2Pairs, w4dense_fp8),
                          "dflash2-pairs applies and is registered on W4Dense with the DFlash2 drafter");
        q::FastPathSelection w4{.configuration = w4dense_fp8, .paths = {FastPath::GdnConvFused}};
        failures += check(w4.forward().gdn_conv_fused &&
                              w4.forward().dense_routes == infernix::ops::LinearRoutes::Exact &&
                              w4.request_invariant(),
                          "the fused convolution alone (as a candidate) sets its own flag and stays reproducible");
        w4.paths = {FastPath::GdnConvFused, FastPath::Q8SlicedK, FastPath::DFlash2Pairs};
        failures += check(w4.forward().gdn_conv_fused &&
                              w4.forward().dense_routes == infernix::ops::LinearRoutes::Q8SlicedK &&
                              !w4.request_invariant(),
                          "q8-sliced-k sets the dense route flag and leaves the reproducible route");
    }
    // f16-accumulate-prefill acts on the 8-bit-code dense projections (recipe C's row FP8, FP4-Local's
    // MXFP8, Dense8's Q8) only, sets its route flag, couples a prompt to its call boundaries, and is
    // qualified only where it measured faster (FP4-Local, INT8 KV; the owner's decision, 2026-10-25).
    for (const DenseKernels dense : {DenseKernels::Fp8Row, DenseKernels::Mxfp8}) {
        failures += check(q::fast_path_applies(FastPath::F16AccumulatePrefill, {dense, false, KvCacheStorage::Int8Group64},
                                               options),
                          "f16-accumulate-prefill runs on FP8 and MXFP8 dense weights");
    }
    for (const DenseKernels dense : {DenseKernels::Bf16, DenseKernels::Ggml, DenseKernels::Nvfp4, DenseKernels::Q8}) {
        failures += check(!q::fast_path_applies(FastPath::F16AccumulatePrefill, {dense, false, KvCacheStorage::Int8Group64},
                                                options),
                          "f16-accumulate-prefill has no route for 16-bit, GGUF or Q8 dense weights");
    }
    failures += check(!infernix::ops::has_route(exact.prefill_routes, infernix::ops::LinearRoutes::F16Accumulate),
                      "the exact route keeps FP8 and MXFP8 projections exact");
    selection.paths = {FastPath::F16AccumulatePrefill};
    failures += check(selection.forward().prefill_routes == infernix::ops::LinearRoutes::F16Accumulate &&
                          selection.forward().dense_routes == infernix::ops::LinearRoutes::Exact,
                      "f16-accumulate-prefill sets its route flag for prefill calls only");
    failures += check(!selection.request_invariant(), "f16-accumulate-prefill couples a prompt to its call boundaries");
    for (const DenseKernels dense : {DenseKernels::Fp8Row, DenseKernels::Mxfp8}) {
        for (const bool a4 : {false, true}) {
            for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                  KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                  KvCacheStorage::Q4KeyVq2Value}) {
                // No configuration registers it on GB10: measured against the retuned exact route it is
                // 5.3-36.2 % slower at its 256-column admission on four of the five MXFP8 problems, and
                // faster only on [13952,2560] from 4096 columns (docs/gb10/20261029-changelog.md).
                failures += check(!q::fast_path_qualified(FastPath::F16AccumulatePrefill, {dense, a4, kv}),
                                  "f16-accumulate-prefill registers no configuration on GB10");
            }
        }
    }

    // screened-lm-head screens generation's BF16 and 8-bit heads (a GGUF head has no exact rescore), is
    // qualified only where it measured faster (FP4-Local, INT8 KV; the owner's decision, 2026-10-25), couples
    // a request to the rows of its rounds, and changes no Forward flag (the Program decides it per round).
    for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Mxfp8}) {
        failures += check(q::fast_path_applies(FastPath::ScreenedLmHead, {dense, false, KvCacheStorage::Int8Group64}, options),
                          "screened-lm-head applies to generation with a non-GGUF head");
    }
    failures += check(!q::fast_path_applies(FastPath::ScreenedLmHead, {kGgml, false, KvCacheStorage::Int8Group64}, options),
                      "screened-lm-head has no exact rescore for a GGUF head");
    {
        infernix::EngineOptions scoring;
        scoring.purpose = infernix::EnginePurpose::CausalScoring;
        failures += check(!q::fast_path_applies(FastPath::ScreenedLmHead, dense8, scoring),
                          "screened-lm-head does not apply to causal scoring");
    }
    for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Mxfp8,
                                     DenseKernels::Nvfp4}) {
        for (const bool a4 : {false, true}) {
            for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
                                  KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                  KvCacheStorage::Q4KeyVq2Value}) {
                const bool measured = (dense == DenseKernels::Mxfp8 && !a4 && kv == KvCacheStorage::Int8Group64) ||
                                      (dense == DenseKernels::Nvfp4 && a4);
                failures += check(q::fast_path_qualified(FastPath::ScreenedLmHead, {dense, a4, kv}) == measured,
                                  "screened-lm-head is qualified on FP4-Local (INT8 KV) and W4Dense");
            }
        }
    }
    selection.paths = {FastPath::ScreenedLmHead};
    failures += check(!selection.request_invariant(), "screened-lm-head couples a request to its rounds' rows");
    failures += check(selection.forward().dense_routes == infernix::ops::LinearRoutes::Exact,
                      "screened-lm-head sets no dense route");
    failures += check(infernix::parse_fast_path("screened-lm-head") == FastPath::ScreenedLmHead,
                      "screened-lm-head parses");

    // fp16-gdn-state applies to every artifact (36 GDN layers each), is qualified on W4Dense (Nvfp4, a4=true),
    // keeps a request invariant (its roundings follow the request's own calls) and sets no Forward flag (the
    // Program's state pool carries it).
    for (const DenseKernels dense : {DenseKernels::Q8, DenseKernels::Bf16, DenseKernels::Fp8Row, DenseKernels::Mxfp8,
                                     DenseKernels::Ggml, DenseKernels::Nvfp4}) {
        failures += check(q::fast_path_applies(FastPath::Fp16GdnState, {dense, false, KvCacheStorage::Int8Group64}, options) &&
                              !q::fast_path_qualified(FastPath::Fp16GdnState, {dense, false, KvCacheStorage::Int8Group64}),
                          "fp16-gdn-state is an unqualified candidate on a4=false artifacts");
    }
    failures += check(q::fast_path_qualified(FastPath::Fp16GdnState, {DenseKernels::Nvfp4, true, KvCacheStorage::Int8Group64}),
                      "fp16-gdn-state is qualified on W4Dense");
    selection.paths = {FastPath::Fp16GdnState};
    failures += check(selection.request_invariant(), "fp16-gdn-state coupled requests");
    failures += check(selection.forward().dense_routes == infernix::ops::LinearRoutes::Exact,
                      "fp16-gdn-state sets a dense route");
    failures += check(infernix::parse_fast_path("fp16-gdn-state") == FastPath::Fp16GdnState, "fp16-gdn-state parses");

    // block-verification: DFlash2 and MTP chains (the program accepts them through the sparse rule),
    // qualified on W4Dense (Nvfp4, a4=true), and a row's acceptance reads only that row.
    {
        infernix::EngineOptions dflash2 = options;
        dflash2.speculative.backend     = infernix::SpeculativeBackend::DFlash2;
        infernix::EngineOptions mtp     = options;
        mtp.speculative.backend         = infernix::SpeculativeBackend::Mtp;
        infernix::EngineOptions none    = options;
        none.speculative.backend        = infernix::SpeculativeBackend::None;
        const FastPathConfiguration dense{DenseKernels::Q8, false, KvCacheStorage::Int8Group64};
        const FastPathConfiguration w4dense{DenseKernels::Nvfp4, true, KvCacheStorage::Int8Group64};
        failures += check(q::fast_path_applies(FastPath::BlockVerification, dense, dflash2) &&
                              q::fast_path_applies(FastPath::BlockVerification, dense, mtp) &&
                              !q::fast_path_applies(FastPath::BlockVerification, dense, none) &&
                              !q::fast_path_qualified(FastPath::BlockVerification, dense) &&
                              q::fast_path_qualified(FastPath::BlockVerification, w4dense),
                          "block-verification is qualified on W4Dense and an unqualified candidate on a4=false drafted chains");
        selection.paths = {FastPath::BlockVerification};
        failures += check(selection.request_invariant(), "block-verification coupled requests");
    }

    // --a8-activations' routes (A8Routes): decode routes on a GGUF import's dense projections and on GGUF and
    // NVFP4 W4A16 experts (not W4A4 or block-FP8 ones, nor beside expert-a4-residual-decode's two A4 terms),
    // prefill routes on GGUF and MXFP8 dense projections and on the same experts (whose wide route needs the
    // narrow one); an artifact with none of them refuses the flag.
    {
        using infernix::ops::LinearRoutes;
        const auto routes = [](DenseKernels dense, bool a4, QType experts, std::size_t ggml_projections,
                               bool a4_narrow = false) {
            return q::a8_activation_routes({dense, a4, KvCacheStorage::Int8Group64}, experts, ggml_projections,
                                           a4_narrow);
        };
        const q::A8Routes iq3s = routes(kGgml, false, QType::GGML_IQ3_S, 120);
        failures += check(iq3s.ggml_projections == 120 && iq3s.ggml_dense && iq3s.ggml_experts &&
                              iq3s.ggml_dense_prefill && iq3s.experts_prefill && !iq3s.nvfp4_experts &&
                              !iq3s.mxfp8_dense,
                          "a GGUF import takes its dense projections' and experts' decode and prefill routes");
        const q::A8Routes fp4_local = routes(DenseKernels::Mxfp8, false, QType::NVFP4_MUL, 0);
        failures += check(fp4_local.nvfp4_experts && fp4_local.experts_prefill && fp4_local.mxfp8_dense &&
                              !fp4_local.ggml_experts && !fp4_local.ggml_dense && !fp4_local.ggml_dense_prefill,
                          "FP4-Local takes its W4A16 experts' decode and prefill routes and its MXFP8 dense prefill route");
        const q::A8Routes fp4_local_a4 = routes(DenseKernels::Mxfp8, true, QType::NVFP4_MUL, 0);
        failures += check(fp4_local_a4.any() && !fp4_local_a4.nvfp4_experts && !fp4_local_a4.experts_prefill &&
                              fp4_local_a4.mxfp8_dense,
                          "FP4-Local with 4-bit expert activations takes the MXFP8 prefill route only");
        const q::A8Routes recipe_c_a8 = routes(DenseKernels::Fp8Row, false, QType::NVFP4_MUL, 0);
        failures += check(recipe_c_a8.nvfp4_experts && recipe_c_a8.experts_prefill && !recipe_c_a8.mxfp8_dense,
                          "recipe C takes its W4A16 experts' routes (its FP8 dense prefill is fp8-a8-pair's)");
        failures += check(!routes(DenseKernels::Fp8Row, false, QType::NVFP4_MUL, 0, true).any(),
                          "expert-a4-residual-decode's two A4 terms keep the W4A16 experts' narrow calls, and so "
                          "their wide calls off the 8-bit route");
        struct Refused {
            DenseKernels dense;
            bool a4;
            QType experts;
        };
        for (const Refused& refused : {Refused{DenseKernels::Q8, true, QType::NVFP4_MUL},
                                       Refused{kBf16, false, QType::FP8_E4M3FN_B128},
                                       Refused{kBf16, true, QType::NVFP4_MUL}}) {
            failures += check(!routes(refused.dense, refused.a4, refused.experts, 0).any(),
                              "Dense8, the official FP8 import and recipe A have no 8-bit activation route");
        }
        // The decode flags; the MXFP8 pair replaces f16-accumulate-prefill, in prefill calls only.
        auto fp4_local_fast =
            q::FastPathSelection{.configuration = {DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64},
                                 .paths         = {FastPath::F16AccumulatePrefill}}
                .forward();
        fp4_local.apply(fp4_local_fast);
        failures += check(fp4_local_fast.prefill_routes == LinearRoutes::Fp8A8PairBlock32 &&
                              fp4_local_fast.dense_routes == LinearRoutes::Exact && fp4_local_fast.a8_experts &&
                              fp4_local_fast.a8_experts_wide,
                          "--a8-activations on FP4-Local: the block32 pair in place of the FP16 prefill route");
        auto iq3s_fast = q::FastPathSelection{.configuration = {kGgml, false, KvCacheStorage::Int8Group64},
                                              .paths         = {FastPath::GgmlBf16Prefill}}
                             .forward();
        iq3s.apply(iq3s_fast);
        failures += check(iq3s_fast.dense_routes == LinearRoutes::GgmlA8 &&
                              iq3s_fast.prefill_routes == (LinearRoutes::GgmlBf16Wide | LinearRoutes::GgmlA8Wide) &&
                              iq3s_fast.a8_experts && iq3s_fast.a8_experts_wide,
                          "--a8-activations on a GGUF import: GgmlA8 on every call, GgmlA8Wide (ahead of the BF16 "
                          "tiles) on prefill calls, the experts' 8-bit routes");
        failures += check(fp4_local_a4.describe().find("decode and verification: none") != std::string::npos &&
                              recipe_c_a8.describe().find("prefill: the routed experts") != std::string::npos,
                          "the startup line names what takes 8-bit activations in decode and in prefill");
    }

    // univer-trees: applies to generation with MTP or DFlash2 and scalar KV storage (excluding VQ
    // storages and None), qualified on W4Dense (Nvfp4, a4=true, scalar KV), request-invariant.
    {
        infernix::EngineOptions mtp_opt = options;
        mtp_opt.speculative.backend     = infernix::SpeculativeBackend::Mtp;
        infernix::EngineOptions df_opt  = options;
        df_opt.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        infernix::EngineOptions none_opt = options;
        none_opt.speculative.backend     = infernix::SpeculativeBackend::None;
        infernix::EngineOptions score_opt = mtp_opt;
        score_opt.purpose                 = infernix::EnginePurpose::CausalScoring;

        const FastPathConfiguration w4_int8{DenseKernels::Nvfp4, true, KvCacheStorage::Int8Group64};
        const FastPathConfiguration w4_k8v4{DenseKernels::Nvfp4, true, KvCacheStorage::Fp8KeyNvfp4Value};
        const FastPathConfiguration w4_vq2{DenseKernels::Nvfp4, true, KvCacheStorage::Vq2};
        const FastPathConfiguration w4_k4v2{DenseKernels::Nvfp4, true, KvCacheStorage::Q4KeyVq2Value};
        const FastPathConfiguration a4_false{DenseKernels::Mxfp8, false, KvCacheStorage::Int8Group64};

        failures += check(q::fast_path_applies(FastPath::UniverTrees, w4_int8, mtp_opt) &&
                              q::fast_path_applies(FastPath::UniverTrees, w4_k8v4, df_opt) &&
                              !q::fast_path_applies(FastPath::UniverTrees, w4_int8, none_opt) &&
                              !q::fast_path_applies(FastPath::UniverTrees, w4_int8, score_opt) &&
                              !q::fast_path_applies(FastPath::UniverTrees, w4_vq2, mtp_opt) &&
                              !q::fast_path_applies(FastPath::UniverTrees, w4_k4v2, df_opt),
                          "univer-trees applies only to scalar-KV generation with MTP or DFlash2");
        failures += check(q::fast_path_qualified(FastPath::UniverTrees, w4_int8) &&
                              q::fast_path_qualified(FastPath::UniverTrees, w4_k8v4) &&
                              !q::fast_path_qualified(FastPath::UniverTrees, w4_vq2) &&
                              !q::fast_path_qualified(FastPath::UniverTrees, a4_false),
                          "univer-trees is qualified on W4Dense with scalar KV formats only");
        selection.configuration = w4_int8;
        selection.paths         = {FastPath::UniverTrees};
        failures += check(selection.request_invariant(), "univer-trees stays request-invariant");
        failures += check(infernix::parse_fast_path("univer-trees") == FastPath::UniverTrees, "univer-trees parses");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Qwen3.8-Flash-Next fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
