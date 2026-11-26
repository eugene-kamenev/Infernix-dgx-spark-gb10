#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/temporary_switches.h"

#include "core/vram_budget.h"
#include "models/load_options.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/measurement.h"
#include "runtime/expert_offload/memory_plan.h"
#include "models/qwen4_exp/program/fast_paths.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "runtime/expert_offload/route_trace.h"
#include "runtime/expert_offload/vram_monitor.h"
#include "models/registry.h"
#include "runtime/engine/expert_memory.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace infernix::runtime {
namespace {

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

// The fixed device allocations, largest first in the order a user can act on them.
std::string device_contributors(std::uint64_t dense, const models::qwen4_exp::ProgramDevicePlan& plan) {
    char kv[64];
    if (plan.kv_max > plan.kv) {
        std::snprintf(kv, sizeof(kv), "%llu (grows to %llu from the expert cache)", mib(plan.kv), mib(plan.kv_max));
    } else {
        std::snprintf(kv, sizeof(kv), "%llu", mib(plan.kv));
    }
    char text[320];
    std::snprintf(text, sizeof(text),
                  "dense weights %llu MiB, KV %s, workspace %llu, expert staging %llu, state and io %llu, expert "
                  "tables %llu",
                  mib(dense), kv, mib(plan.workspace), mib(plan.staging), mib(plan.state + plan.io),
                  mib(plan.residency));
    return text;
}

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    report_engine(options, level, text);
}

std::filesystem::path default_dflash2_model(const std::filesystem::path& artifact_path) {
    std::error_code ec;
    const std::filesystem::path sibling(artifact_path.string() + ".dflash2");
    if (std::filesystem::is_regular_file(sibling, ec)) { return sibling; }
    const std::filesystem::path dir = artifact_path.parent_path();
    if (!dir.empty()) {
        const std::filesystem::path named = dir / "Qwen3.8-Flash-Next-DFlash2-Infernix.infernix";
        if (std::filesystem::is_regular_file(named, ec)) { return named; }
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file(ec)) { continue; }
            const std::string name = entry.path().filename().string();
            if ((entry.path().extension() == ".infernix" || entry.path().extension() == ".ninfer") &&
                (name.find("DFlash2") != std::string::npos || name.find("dflash2") != std::string::npos)) {
                return entry.path();
            }
        }
        const std::filesystem::path neighbor =
            dir.parent_path() / "Qwen3.8-Flash-Next-DFlash2-Head" / "Qwen3.8-Flash-Next-DFlash2-Infernix.infernix";
        if (std::filesystem::is_regular_file(neighbor, ec)) { return neighbor; }
    }
    return {};
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation && options.purpose != EnginePurpose::CausalScoring) {
        throw std::invalid_argument("Qwen3.8-Flash-Next serves generation and causal scoring");
    }
    if (options.speculative.backend == SpeculativeBackend::DFlash || options.speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next speculates with its MTP drafter, a DFlash2 drafter "
                                    "(--spec dflash2 --draft-model FILE) and n-gram copy proposals");
    }
    // Verification widths up to 8 keep its dense routes column-invariant (design 11.3); its rounds
    // verify chains.
    if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
        (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 7)) {
        throw std::invalid_argument("Qwen3.8-Flash-Next DFlash2 drafts 1-7 tokens per round (--draft-tokens)");
    }
    if (options.speculative.draft_tree_auto ||
        std::any_of(options.speculative.draft_tree_nodes.begin(), options.speculative.draft_tree_nodes.end(),
                    [](std::uint32_t nodes) { return nodes != 0; })) {
        throw std::invalid_argument("Qwen3.8-Flash-Next verifies chains: --draft-tree-nodes is not supported");
    }
    // The lossless route (TextConfig::lossless; docs/maintainer/lossless-route.md §2a, §3a): every
    // operation correctly rounded; causal scoring takes each window in one call (its prefill chunk),
    // generation runs text through bf16 KV (V stored as BF16) without speculation.
    if (options.lossless) {
        if (options.fast_paths || !options.fast_path_candidates.empty() ||
            (options.expert_a4_activations && !options.reference) || options.a8_activations) {
            throw std::invalid_argument("the lossless route runs no fast path: disable fast paths");
        }
        if (options.purpose == EnginePurpose::CausalScoring && options.prefill_chunk < options.max_context) {
            throw std::invalid_argument("the lossless route scores each window in one call: --prefill-chunk must cover "
                                        "--max-context");
        }
        if (options.enable_vision || options.speculative.backend != SpeculativeBackend::None ||
            options.speculative.ngram_draft_tokens != 0 || !options.speculative.draft_model.empty()) {
            throw std::invalid_argument("the lossless route takes text without speculation or the vision tower");
        }
        if (options.purpose == EnginePurpose::Generation &&
            (options.kv_cache != KvCacheStorage::BFloat16 || options.expert_tail_skip != 0)) {
            throw std::invalid_argument("lossless generation takes bf16 KV and no --expert-tail-skip");
        }
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.expert_tail_skip > 9) {
        throw std::invalid_argument("--expert-tail-skip takes a top-10 rank 1-9 (0: off)");
    }
    if (options.expert_tail_skip != 0 && options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("--expert-tail-skip applies to generation (its decode rounds) only");
    }
}

} // namespace

bool artifact_is_qwen4_exp(const artifact::Reader& reader) { return models::qwen4_exp::is_qwen4_exp(reader); }

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path      = options.chat_template_path,
                                               .architecture            = models::Architecture::Qwen4Exp,
                                               .vision_enabled          = options.enable_vision,
                                               .max_context             = options.max_context,
                                               .media_cache_bytes       = options.media_cache_bytes,
                                               .media_live_bytes        = options.media_live_bytes,
                                               .media_preprocess_threads = options.media_preprocess_threads,
                                               .vision_max_merged_tokens = options.vision_max_merged_tokens,
                                               .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
                                               .thinking_budget_message  = options.thinking_budget_message})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) {
        throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM");
    }
    inspect.complete();
    // DEFAULT ON an integrated GPU (GB10): the routed expert banks land in device memory through the
    // ordinary weight upload instead of being pinned in host memory. On unified DRAM that is one draw
    // per byte instead of two (pinned bank plus its frame copies), nothing has to be streamed or
    // staged, and the kernels read records straight from their final home.
    const bool integrated = open_vram_budget_source(device.device)->query().integrated;
    bool device_experts   = integrated; // the default: on-device experts exactly where they pay
    if (const char* env = std::getenv("INFERNIX_EXPERTS_ON_GPU")) {
        if (std::atoi(env) == 0) { // explicit opt-out -> the legacy pinned-host path
            device_experts = false;
            report(options, DiagnosticLevel::Info,
                   "INFERNIX_EXPERTS_ON_GPU=0: using the pinned-host expert path");
        } else if (!integrated) { // forced on where it cannot apply
            throw std::runtime_error("INFERNIX_EXPERTS_ON_GPU=1 needs an integrated (unified-memory) GPU");
        }
    }
    if (device_experts && options.enable_vision && options.vision_offload == VisionOffload::On) {
        report(options, DiagnosticLevel::Info,
               "experts-on-GPU keeps vision weights on device instead of offloading them");
    }
    const auto qwen4_load = [&](bool stream_experts) {
        auto load           = models::load_options(options, models::Architecture::Qwen4Exp);
        load.stream_experts = stream_experts;
        load.device_experts = device_experts;
        if (device_experts) { load.vision_offload = false; }
        if (integrated) { load.mtp_expert_pool = false; }
        if (load.dflash2() && load.draft_model.empty()) {
            load.draft_model = default_dflash2_model(options.artifact_path);
        }
        return load;
    };
    auto plan = models::qwen4_exp::plan_load(reader, qwen4_load(false));
    // The RAM ledger (design §19.3.7): the experts take what the reserve and every other planned
    // allocation leave, before anything is pinned or read. It is planned below with the Program's options.
    auto vram = open_vram_budget_source(device.device);
    // --host-context-mib auto: the prefix cache's Host tier is sized after the experts and the n-gram table.
    const bool prefix_auto     = options.context_cache.enabled && options.context_cache.host_capacity_auto;
    std::uint64_t prefix_bytes = options.context_cache.enabled && !prefix_auto ? options.context_cache.host_capacity_bytes.value() : 0;
    auto demand = expert_ram_demand(
        options, {.expert_banks = device_experts ? 0 : plan.pinned_expert_bytes(), .other_pinned = plan.pinned_other_bytes()},
        prefix_bytes, vram->query());
    // The n-gram table its volume holds: the artifact's, or another volume's (any format) in its place.
    const std::filesystem::path volume = options.ngram_volume_path.empty()
                                             ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                             : options.ngram_volume_path;
    const models::qwen4_exp::NgramTableConfig table =
        models::qwen4_exp::resolve_ngram_volume(volume, plan.config().text.ple);
    if (const auto& own = plan.config().text.ple.table;
        table.format != own.format || table.volume_id != own.volume_id) {
        const auto name = [](models::qwen4_exp::NgramRowFormat format) {
            return format == models::qwen4_exp::NgramRowFormat::Bf16    ? std::string("BF16")
                   : format == models::qwen4_exp::NgramRowFormat::Iq4Nl ? std::string("IQ4_NL")
                   : format == models::qwen4_exp::NgramRowFormat::Nvfp4 ? std::string("NVFP4")
                                                                        : std::string("FP8");
        };
        report(options, DiagnosticLevel::Info,
               "n-gram table: " + name(table.format) + " rows from " + volume.string() + " in place of the artifact's " +
                   name(own.format) + " table");
    }
    demand.pageable = models::qwen4_exp::NgramVolume::cache_bytes(table, options.ngram_cache_bytes);

    models::qwen4_exp::ProgramOptions program_options;
    program_options.device_experts = device_experts;
    program_options.unified_memory = integrated;
    program_options.use_cuda_graph = options.use_cuda_graph;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.causal_scoring  = options.purpose == EnginePurpose::CausalScoring;
    program_options.prefill_chunk   = options.prefill_chunk;
    // The lossless route reads its experts through serial staging passes: no prefill stream, layer
    // walk or CPU split.
    if (options.lossless) {
        program_options.prefill_stream    = false;
        program_options.prefill_cpu_split = false;
    }
    program_options.kv_cache        = options.kv_cache;
    program_options.diagnostics     = options.diagnostic_observer;
    program_options.kv_capacity_tokens =
        options.kv_capacity.mode == KvCapacityMode::Explicit ? options.kv_capacity.explicit_tokens : 0U;
    program_options.ngram_draft_tokens = options.speculative.ngram_draft_tokens;
    program_options.ngram_min_match    = options.speculative.ngram_min_match;
    program_options.mtp_draft_tokens   = options.speculative.backend == SpeculativeBackend::Mtp
                                             ? options.speculative.draft_tokens
                                             : 0U;
    program_options.dflash2_draft_tokens = options.speculative.backend == SpeculativeBackend::DFlash2
                                               ? options.speculative.draft_tokens
                                               : 0U;
    // A fixed draft length verifies every draft unless --spec-min-p asks for the cut.
    program_options.auto_draft_length = options.speculative.draft_tokens_auto;
    // MTP's automatic length cuts by its drafts' cumulative acceptance (MtpCumulative) and DFlash2's
    // calibrated length needs no cut.
    program_options.draft_min_p = options.speculative.draft_min_p.value_or(0.0F);
    if (!options.expert_state_path.empty()) {
        // The state belongs to the artifact: its experts and their records, not the build or KV format.
        std::error_code error;
        const auto size = std::filesystem::file_size(options.artifact_path, error);
        const auto time = std::filesystem::last_write_time(options.artifact_path, error);
        program_options.expert_state          = options.expert_state_path;
        program_options.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                                ";size=" + std::to_string(error ? 0U : size) +
                                                ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    }
    program_options.expert_seed = !options.expert_state_off;
    program_options.ngram_volume = volume;
    program_options.ngram_table  = table;
    // The hybrid prefix cache (design §19.3.1). Host-born snapshots live in the pinned slab pool;
    // the cost model ranks resume sources by the measured prefill coefficients.
    const ContextCacheOptions& cache = options.context_cache;
    if (cache.enabled) {
        const std::uint32_t chunk         = options.prefill_chunk;
        program_options.prefix_cache      = true;
        // normalize_engine_options resolved every value for this architecture but an automatic Host tier's size,
        // which the RAM plan below sets (prefix_host_bytes).
        program_options.prefix_taps.max_new_taps   = cache.hybrid.max_new_taps.value();
        program_options.prefix_taps.ladder_tokens  = cache.hybrid.tap_ladder_tokens.value();
        program_options.prefix_taps.min_gap_tokens = cache.hybrid.tap_min_gap_tokens.value();
        // Fitted 2026-10-05 (RTX 5090 at PCIe 5.0 x8, INT8 KV; design §19.4): 1.67 s per layer-walk
        // span, 0.16 s per further call, 22 us per token and 1.06 ns per attention pair, within 5 %
        // from 4K to 128K. The span is one pass of the experts over the link: the Program rescales
        // it, and the restore rate, to the link it measures at startup.
        program_options.prefix_cost.chunk_seconds          = integrated ? 0.56 : 0.1645;
        program_options.prefix_cost.chunk_tokens           = chunk;
        program_options.prefix_cost.token_seconds          = integrated ? 9.0e-5 : 2.23e-5;
        program_options.prefix_cost.attention_pair_seconds = integrated ? 4.0e-9 : 1.06e-9;
        program_options.prefix_cost.call_route_fraction    = 10.0 / 512.0;
        program_options.prefix_span_seconds                = device_experts ? 0.0 : 1.67;
        program_options.prefix_cost.h2d_bytes_per_second   = integrated ? 19.4e9 : 27.5e9; // the reference link, until measured
        program_options.prefix_cost.transfer_batch_seconds = 20.0e-6;
        program_options.prefix_save                        = cache.hybrid.persistent_save;
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait is
        // kept well inside its queue timeout.
        program_options.prefix_coalesce_wait_seconds = static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0;
    }
    program_options.route_trace   = options.route_trace_path.empty()
                                        ? runtime::expert_offload::testing::route_trace()
                                        : options.route_trace_path;
    program_options.vram_grow_delay_seconds = runtime::expert_offload::testing::vram_grow_delay();
    const std::optional<std::uint64_t> effective_vram_headroom =
        options.vram_headroom_bytes.has_value()
            ? std::optional<std::uint64_t>(*options.vram_headroom_bytes)
            : (integrated ? std::optional<std::uint64_t>(0) : std::nullopt);
    program_options.vram_headroom = effective_vram_headroom;
    program_options.expert_tail_skip = options.expert_tail_skip;

    if (options.expert_tail_skip != 0) {
        report(options, DiagnosticLevel::Warning,
               "--expert-tail-skip " + std::to_string(options.expert_tail_skip) +
                   ": decode and verification rounds skip a routed expert missing from VRAM that every token "
                   "routes at top-10 rank " + std::to_string(options.expert_tail_skip) +
                   " or lower; output changes (lower quality) for faster decode");
    }

    // The VRAM check (design §19.3.7), before the weights are read: the dense weights, the
    // Program's fixed allocations, the reserve for graph executables and the display headroom
    // must leave room for the expert cache. Fast paths are selected once the weights are bound, so
    // the check counts sampled-mtp-drafts' draft graphs whenever it could be selected.
    program_options.sampled_mtp_drafts =
        options.speculative.backend == SpeculativeBackend::Mtp &&
        (options.fast_paths || std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                         FastPath::SampledMtpDrafts) != options.fast_path_candidates.end());
    // screened-lm-head's graphs likewise: asked for as a candidate, or with fast paths where the loader bound the
    // LM head's screen (the path is qualified per configuration).
    program_options.screened_lm_head =
        std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                  FastPath::ScreenedLmHead) != options.fast_path_candidates.end() ||
        (options.fast_paths && plan.has_output_head_screen());
    // fp16-gdn-state / bf16-gdn-state's smaller pool likewise.
    program_options.fp16_gdn_state = std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                               FastPath::Fp16GdnState) != options.fast_path_candidates.end();
    program_options.bf16_gdn_state = std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(),
                                               FastPath::Bf16GdnState) != options.fast_path_candidates.end();

    // The frame cache duplicates records that are already on device; it exists to hold experts
    // the pinned pool could not, and draws up to ~25-32 GiB of extra unified DRAM on an
    // integrated GPU. Refuse rather than silently wedge the box when both are requested.
    if (device_experts && options.expert_ram_bytes.has_value()) {
        throw std::invalid_argument(
            "--expert-ram-mib conflicts with experts-on-GPU (the default on integrated GPUs): the "
            "experts sit on device whole, so the RAM tier and its frame cache have nothing to do");
    }

    // ---- The RAM plan: the experts, then the n-gram table, then the prefix cache's Host tier ----
    // --expert-ram-layout full: every expert keeps a RAM copy (full mode) whenever the ledger allows; an explicit
    // --expert-ram-mib caps it. exclusive (design §19.3.7): RAM holds the experts VRAM does not plus shadows for what
    // VRAM gives back without a promotion: a prefill's loan (its wide arena and ring, measured at a quarter of the
    // frames: 2,901-3,098 of 12,576 on IQ3_S at 8K) and the KV growth that takes frames. The warm start then seeds the
    // rest of the frames, and the fill phase's landings keep their RAM copies as those shadows; the loan and the
    // shrink evict shadowed experts first (ExpertResidency::lend), so nothing leaves both tiers. What it frees stays
    // for the n-gram table (its auto rule) and the Host tier.
    const auto device_plan =
        models::qwen4_exp::Program::plan_device(program_options, plan.config(), plan.public_token_count());
    const std::uint64_t dense = plan.device_bytes();
    const std::uint64_t device_claim =
        integrated ? dense + device_plan.fixed_bytes() +
                         expert_offload::internal_reserve(device_plan.graph_bound,
                                                          expert_offload::kGraphExecutableBytes, false) +
                         effective_vram_headroom.value_or(0)
                   : 0;
    const auto snapshot = host_memory_snapshot();
    bool exclusive      = false;
    if (!device_experts && options.expert_ram_layout.value_or(ExpertRamLayout::Exclusive) == ExpertRamLayout::Exclusive &&
        !options.expert_ram_bytes) {
        const auto& estimate = device_plan;
        expert_offload::VramDemand frames_demand;
        frames_demand.headroom    = effective_vram_headroom;
        frames_demand.elastic     = true;
        frames_demand.graphs      = estimate.graph_bound;
        frames_demand.fixed       = dense + estimate.fixed_bytes();
        frames_demand.frame_bytes = estimate.expert_frame_bytes;
        frames_demand.max_frames  = estimate.max_frames;
        const auto sizing           = expert_offload::size_expert_frames(vram->query(), frames_demand);
        const std::uint64_t frames  = sizing.frames;
        const std::uint64_t keys    = static_cast<std::uint64_t>(plan.config().text.num_hidden_layers) * plan.config().text.moe.experts;
        // A quarter of the frames is the loan; the shadows also cover the copies decode's swaps hold at that moment
        // (a quarter alone left IQ3_S ~570 loan victims a request without a copy: 2026-10-27, x2_iq3_dec8k), so 5/16.
        const std::uint64_t kv_frames =
            estimate.kv_max > estimate.kv && estimate.expert_frame_bytes != 0
                ? (estimate.kv_max - estimate.kv + estimate.expert_frame_bytes - 1) / estimate.expert_frame_bytes
                : 0;
        const std::uint64_t reserve = frames * 5 / 16 + kv_frames;
        // The tier pools its slots by record size, each layer the same share (HostExpertTier), so the experts' slots
        // cost the layers' mean record (the average frame) and only the lists (ring, prefetch, demotion), which hold
        // no expert, cost the largest record each. A layer walk's span lends the prefill stream its slots (free ones
        // first): without room for them a span evicted that many RAM-only experts, which came back from the SSD
        // (2026-10-28, IQ3_S at 16K: 2,563 stream reads and 833 decode reads a request; 0.89x of the full layout).
        // The reserve is counted twice: until decode fills the frames the warm start left unseeded, their experts take
        // RAM slots as RAM-only residents, while the shadows a prefill's loan needs take the reserve again. With it
        // once, a cold 16K request lost ~1,000 experts to the SSD and decoded 237 tok/s; twice, 0 and 247 (0.5x: 4 and
        // 246; the full layout 254; 2026-10-28, IQ3_S).
        const std::uint64_t experts_slots =
            (keys > frames - std::min(frames, reserve) ? keys - (frames - std::min(frames, reserve)) : 0) + reserve +
            estimate.expert_tier_stream_slots;
        const std::uint64_t bytes = experts_slots * estimate.expert_frame_bytes +
                                    estimate.expert_tier_list_slots * estimate.expert_record_stride;
        // Plus 2 % for the lock overhead the tier takes off its pinned bytes (IQ3_S: 15,255 of 15,468 planned slots
        // pinned, leaving 217 experts SSD-only at the warm start).
        const std::uint64_t cap = bytes + bytes / 50;
        if (frames > reserve && cap + expert_offload::kLedgerGiB < demand.expert_banks) {
            demand.expert_cap = cap;
            exclusive         = !expert_offload::plan_host_memory(snapshot, demand).cap_too_large;
            if (exclusive) {
                // The warm start seeds the full layout's share of the frames (never the loan reserve): the free
                // frames take the fill phase's landings. Seeding every frame measured slower (IQ3_S 16K decode 223
                // against 235 tok/s); seeding all but the reserve (68 % of the frames) left fewer landings (2,878
                // against 4,281 a 16K request) and more CPU-served misses: 16K decode 258.2 / 258.8 against
                // 264.2 / 264.6 tok/s at half (full layout 266.5 / 266.4), struct 455.0 / 455.5 against 457.7 /
                // 457.4 (full 463.1 / 462.9), the same 47.4 GiB resident (2026-10-29).
                const double reserve_share          = static_cast<double>(reserve) / static_cast<double>(frames);
                program_options.expert_seed_share   = std::min(program_options.expert_seed_share, 1.0 - reserve_share);
                program_options.expert_shadow_share = reserve_share;
                program_options.expert_ram_exclusive = true;
            } else {
                demand.expert_cap.reset(); // a machine too small for it: the ledger's own share (the SSD tier)
            }
        }
    }
    // The n-gram table in RAM (design §12.5): read at Program start into memory locked in the working set for the
    // CPU, which alone reads it, so it takes physical RAM but no share of the page-locking limit (pin room) the
    // experts need. Auto takes it only while the experts keep their planned RAM beside it (every expert in full mode)
    // and --ram-free-mib stays free beyond the Host tier's floor; Ram takes it whatever the experts' placement.
    bool ngram_in_ram = false;
    if (options.ngram_residency != NgramResidency::Volume) {
        auto with_table         = demand;
        with_table.pageable     = models::qwen4_exp::NgramVolume::resident_bytes(table); // in place of the row cache
        auto without_trial      = demand;
        auto with_table_trial   = with_table;
        without_trial.reserve += device_claim;
        with_table_trial.reserve += device_claim;
        const auto without      = expert_offload::plan_host_memory(snapshot, without_trial);
        const auto trial        = expert_offload::plan_host_memory(snapshot, with_table_trial);
        const bool experts_kept = demand.expert_cap ? trial.expert_ram >= without.expert_ram
                                                    : trial.placement == expert_offload::ExpertPlacement::Full;
        const std::uint64_t tier_floor =
            prefix_auto ? ((integrated && !device_experts) ? kDefaultHybridHostCacheBytes : kMinAutoHostContextBytes) : 0;
        if (options.ngram_residency == NgramResidency::Ram ||
            (experts_kept && !trial.cap_too_large && ledger_left_bytes(trial) >= tier_floor)) {
            demand       = with_table;
            ngram_in_ram = true;
        }
    }
    report(options, DiagnosticLevel::Debug,
           std::string("n-gram table: ") +
               (ngram_in_ram ? "in RAM (" + std::to_string(models::qwen4_exp::NgramVolume::resident_bytes(table) >> 20) +
                                   " MiB, locked for the CPU)"
                             : options.ngram_residency == NgramResidency::Volume
                                   ? "read from its volume"
                                   : "read from its volume (the RAM ledger keeps the experts resident first)"));
    // --host-context-mib auto: the Host tier from what the experts and the table leave beyond --ram-free-mib.
    if (prefix_auto) {
        auto prefix_trial = demand;
        prefix_trial.reserve += device_claim;
        prefix_bytes        = auto_host_context_bytes(ledger_left_bytes(expert_offload::plan_host_memory(snapshot, prefix_trial)));
        demand.prefix_cache = prefix_bytes;
    }
    if (cache.enabled) { program_options.prefix_host_bytes = prefix_bytes; }
    // The SSD tier (design §19.3.7): the banks stay in the artifact; the Program pins RAM slots for
    // the experts the ledger leaves room for (less their lock overhead) and reads the rest on use.
    const ExpertRamPlan ram = plan_expert_ram(options, demand, snapshot);
    const bool tier         = !device_experts && ram.tier;
    if (tier) { plan = models::qwen4_exp::plan_load(reader, qwen4_load(true)); }
    plan.set_host_reserve(options.ram_free_bytes, demand.later_pinned);
    program_options.ngram_in_ram      = ngram_in_ram;
    program_options.ngram_cache_bytes = options.ngram_cache_bytes;
    if (!options.expert_stripe_path.empty()) {
        if (tier) {
            program_options.expert_stripe = options.expert_stripe_path;
        } else {
            report(options, DiagnosticLevel::Info,
                   "--expert-stripe is not used: every expert fits in RAM, so none is read from disk");
        }
    }
    // On a unified-memory card the expert frames, this process's Host allocations and the RAM reserve
    // all draw on one physical pool: the engine charges them once and hands the same number to the
    // Program, whose frames are sized from the pool as a whole (memory_plan.h shared_host_claim).
    // The expert bytes pinned on the Host side are the tier's RAM slots, or every bank in full mode;
    // none when the banks sit on device.
    const std::uint64_t host_expert_bytes =
        device_experts
            ? 0ULL
            : tier ? ram.expert_ram_bytes
                   : demand.expert_banks +
                         static_cast<std::uint64_t>(demand.expert_banks * demand.lock_overhead) + 1ULL;
    const std::uint64_t shared_host_bytes = expert_offload::shared_host_claim(demand, host_expert_bytes);
    program_options.shared_host_bytes     = shared_host_bytes;
    if (tier) {
        program_options.expert_ram_bytes      = ram.expert_ram_bytes;
        program_options.expert_unpinned_bytes = ram.expert_unpinned_bytes;
        program_options.later_pinned_bytes    = ram.later_pinned_bytes;
    }
    {
        // The plan in one line, and with each choice's reason at Debug (candidates §22: compare with
        // Basalt's and Strata's totals).
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        auto plan_demand      = demand;
        plan_demand.reserve += device_claim;
        const auto plan_ledger = expert_offload::plan_host_memory(snapshot, plan_demand);
        const double other_gib =
            static_cast<double>(ram.ledger.planned - demand.pageable * (ngram_in_ram ? 1 : 0) - demand.prefix_cache) / kGiB;
        const double free_gib = static_cast<double>(ledger_left_bytes(plan_ledger) + demand.reserve) / kGiB;
        char table_text[64] = "read from its volume";
        if (ngram_in_ram) {
            std::snprintf(table_text, sizeof(table_text), "%.1f GiB",
                          static_cast<double>(models::qwen4_exp::NgramVolume::resident_bytes(table)) / kGiB);
        }
        char prefix_text[64] = "off";
        if (cache.enabled) {
            std::snprintf(prefix_text, sizeof(prefix_text), "%.1f GiB", static_cast<double>(prefix_bytes) / kGiB);
        }
        char line[512];
        std::snprintf(line, sizeof(line),
                      "RAM plan: experts %.1f GiB, n-gram table %s, prefix cache %s, other %.1f GiB; %.1f GiB left free "
                      "for other programs",
                      static_cast<double>(ram.ledger.expert_ram) / kGiB, table_text, prefix_text, other_gib, free_gib);
        report(options, DiagnosticLevel::Info, line);
        std::snprintf(line, sizeof(line),
                      "RAM plan: experts %.1f GiB (%s), n-gram table %.1f GiB (%s), prefix cache %.1f GiB (%s), other %.1f "
                      "GiB; %.1f GiB left free for other programs (--ram-free-mib %.1f)",
                      static_cast<double>(device_experts ? 0ULL : ram.ledger.expert_ram) / kGiB,
                      device_experts ? "on device" : !tier ? "every expert" : exclusive ? "those not in VRAM, and shadows for a prefill's loan" : "the SSD tier",
                      ngram_in_ram ? static_cast<double>(models::qwen4_exp::NgramVolume::resident_bytes(table)) / kGiB : 0.0,
                      ngram_in_ram ? "RAM" : "volume", static_cast<double>(prefix_bytes) / kGiB,
                      !cache.enabled ? "off" : prefix_auto ? "auto" : "set", other_gib, free_gib,
                      static_cast<double>(demand.reserve) / kGiB);
        report(options, DiagnosticLevel::Debug, line);
    }

    const VramSnapshot before = vram->query();
    expert_offload::VramDemand vram_demand;
    vram_demand.headroom    = effective_vram_headroom;
    vram_demand.graphs      = device_plan.graph_bound;
    vram_demand.fixed       = dense + device_plan.fixed_bytes();
    vram_demand.frame_bytes = device_plan.expert_frame_bytes; // an average frame of the per-size pools
    vram_demand.max_frames  = device_plan.max_frames;
    if (before.integrated) {
        vram_demand.frame_pool_cap    = expert_offload::kIntegratedFramePoolCapBytes;
        vram_demand.shared_host_bytes = shared_host_bytes;
    }
    const auto check = expert_offload::size_expert_frames(before, vram_demand);
    if (check.frame_bytes < 0) {
        const std::string pool =
            before.integrated
                ? std::to_string(mib(before.device_free)) + " MiB free plus " + std::to_string(mib(before.reclaimable)) +
                      " MiB the OS can reclaim from this pool ("
                : std::to_string(mib(before.device_free)) + " MiB free (";
        throw std::runtime_error("Qwen3.8-Flash-Next does not fit in device memory: " + pool +
                                 std::to_string(mib(before.device_free + before.reclaimable)) +
                                 " MiB allocatable); " + device_contributors(dense, device_plan) + "; reserve " +
                                 std::to_string(mib(check.reserve)) + "; headroom " + std::to_string(mib(check.headroom)) +
                                 " (" + check.describe() +
                                 "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, "
                                 "lower --vram-headroom-mib or --ram-free-mib, or free memory");
    }
    if (!device_experts && device_plan.max_frames > 0 && check.frames < kFewExpertFrames) {
        report(options, DiagnosticLevel::Warning,
               "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewExpertFrames) + " or more recommended); lower --max-context or --max-concurrency, "
                   "or free VRAM");
    }

    // The dense arena must land in device memory (WDDM can place it in system memory silently).
    SpillGuard guard(*vram);
    guard.begin();
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again (" + guard.readings(dense) + ")");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    // The applicability key of this exact configuration (the architecture dimensions and the bound
    // Text/Vision weight inventory). Serving reports it, an external --context-cost-table and the
    // entries compiled for this machine key on it, and the calibration bench records a measured
    // table under it. Resolution is unconditional: the LoadSummary reports the sources whether or
    // not the prefix cache runs.
    const std::string prefill_signature = models::qwen4_exp::prefill_signature(*instance->model);
    auto context_cost =
        resolve_context_machine_cost({.hardware_class    = context_cost_hardware_class(
                                          device.props.name, device.props.major, device.props.minor),
                                      .prefill_signature = prefill_signature},
                                     options.context_cost.preset_path);
    if (cache.enabled) {
        // A table that binds this configuration prices recomputation with its measured
        // coefficients. The generic estimate is one measured artifact's average, so a
        // configuration no table binds keeps the estimates set above; the Program still replaces
        // the restore rate with the link rate it measures at startup.
        auto priced = context_cost.model;
        if (context_cost.summary.prefill_source == ContextCostPresetSource::GenericDefault) {
            priced.prefill = {};
        }
        layer_hybrid_cost(program_options.prefix_cost, priced);
    }

    // The fast paths qualified for this configuration (docs/maintainer/model-quality.md section 6.7).
    const models::qwen4_exp::FastPathSelection fast = models::qwen4_exp::select_fast_paths(instance->parameters, options);
    program_options.fast_paths        = fast.forward();
    // Two-row speculative rounds: MTP's (adaptive-speculation) or DFlash2's (dflash2-pairs); each
    // applies only to its drafter.
    program_options.pair_speculation  = fast.selected(FastPath::AdaptiveSpeculation) ||
                                        fast.selected(FastPath::DFlash2Pairs);
    program_options.free_cache_resume = fast.selected(FastPath::FreeCacheResume);
    program_options.preemption        = fast.selected(FastPath::Preemption);
    program_options.sampled_mtp_drafts = fast.selected(FastPath::SampledMtpDrafts);
    program_options.screened_lm_head   = fast.selected(FastPath::ScreenedLmHead);
    program_options.fp16_gdn_state     = fast.selected(FastPath::Fp16GdnState);
    program_options.bf16_gdn_state     = fast.selected(FastPath::Bf16GdnState);
    program_options.block_verification = fast.selected(FastPath::BlockVerification);
    program_options.univer_trees =
        fast.selected(FastPath::UniverTrees) && !temporary_switches::flag("INFERNIX_TMP_NO_UNIVER_TREES");
    program_options.gumbel_sampling    = !options.fast_paths;
    program_options.bf16_logits        = options.logits_dtype == LogitsDtype::BF16;
    // --a8-activations (LOSSY, opt-in; design §19.3.31): the 8-bit activation routes of this artifact
    // (models::qwen4_exp::A8Routes), in decode and verification calls and in prefill calls, or a refusal where it
    // has none. Every other call keeps its route.
    if (options.a8_activations) {
        const models::qwen4_exp::A8Routes a8 = models::qwen4_exp::a8_activation_routes(instance->parameters, fast);
        if (!a8.any()) {
            throw std::invalid_argument(
                "--a8-activations: no 8-bit activation route applies to this artifact (it covers a GGUF import's dense "
                "projections and GGUF or NVFP4 W4A16 experts in decode and prefill, and MXFP8 dense projections in "
                "prefill)");
        }
        a8.apply(program_options.fast_paths);
        report(options, DiagnosticLevel::Info,
               "--a8-activations (approximate): " + a8.describe() + "; every other call keeps its route");
    }

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    const auto& sizing = instance->program->vram_sizing();
    const std::uint64_t frames_bytes =
        static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_frame_bytes;
    const std::uint64_t after_startup = vram->query().device_free;
    {
        const char* display = before.display == DisplayState::Headless   ? "no display"
                              : before.display == DisplayState::Attached ? "display attached"
                                                                         : "display unknown";
        char line[704];
        std::snprintf(line, sizeof(line),
                      "VRAM ledger: %llu MiB card, %s, %llu in use before loading; %s; reserve %llu (%u graphs); "
                      "headroom %llu (%s); expert frames %u (%.2f GiB); %llu free after startup",
                      mib(before.device_total), display, mib(before.device_total - before.device_free),
                      device_contributors(dense, device_plan).c_str(), mib(sizing.reserve), device_plan.graph_bound,
                      mib(sizing.headroom), options.vram_headroom_bytes ? "set" : "auto", sizing.frames,
                      static_cast<double>(frames_bytes) / static_cast<double>(1ULL << 30), mib(after_startup));
        report(options, DiagnosticLevel::Debug, line);
        const double gib = static_cast<double>(1ULL << 30);
        char summary[320];
        if (device_experts) {
            std::snprintf(summary, sizeof(summary),
                          "VRAM: model %.2f GiB (experts on device), KV cache %.2f GiB",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib);
        } else if (device_plan.kv_max > device_plan.kv) {
            std::snprintf(summary, sizeof(summary),
                          "VRAM: model %.2f GiB, KV cache %.2f GiB (grows to %.2f GiB as contexts lengthen, taking room "
                          "from the expert cache), expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib,
                          static_cast<double>(device_plan.kv_max) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        } else {
            std::snprintf(summary, sizeof(summary), "VRAM: model %.2f GiB, KV cache %.2f GiB, expert cache %u experts (%.1f GiB)",
                          static_cast<double>(dense) / gib, static_cast<double>(device_plan.kv) / gib, sizing.frames,
                          static_cast<double>(frames_bytes) / gib);
        }
        report(options, DiagnosticLevel::Info, summary);
    }
    const MemorySummary memory = instance->program->memory_summary();
    auto& resolution                         = instance->kv_capacity_resolution;
    resolution.mode                          = options.kv_capacity.mode;
    resolution.main_page_groups              = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups      = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens               = memory.kv_capacity;
    resolution.runtime_reservation_bytes     = device_plan.fixed_bytes() + frames_bytes;
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = after_startup;
    resolution.automatic_headroom_bytes      = sizing.headroom;
    resolution.planned_slack_bytes =
        after_weights > resolution.runtime_reservation_bytes ? after_weights - resolution.runtime_reservation_bytes : 0;
    LoadSummary::PrefixCacheRestore restore;
    if (const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        options.context_cache.enabled && !file.empty()) {
        // Saved states depend on the n-gram rows (another volume's differ from the artifact's table)
        // and on the routed experts' activation precision (--expert-a4-activations).
        std::string signature = "qwen4_exp;ngram=" + std::to_string(static_cast<int>(table.format)) + ":";
        for (const std::uint8_t byte : table.volume_id) {
            constexpr char kHex[] = "0123456789abcdef";
            signature += kHex[byte >> 4U];
            signature += kHex[byte & 15U];
        }
        if (table.fp8_scale) { signature += ";scale=" + std::to_string(*table.fp8_scale); }
        if (table.nvfp4_scale) { signature += ";multiplier=" + std::to_string(*table.nvfp4_scale); }
        signature += instance->model->config().text.moe.a16_experts ? ";experts=a16" : ";experts=a4";
        // A DFlash2 drafter's ring rides in every snapshot: it belongs to that companion artifact.
        if (const auto& draft = instance->model->info().draft_artifact_id) {
            signature += ";draft=";
            for (const std::byte byte : *draft) {
                constexpr char kHex[] = "0123456789abcdef";
                signature += kHex[std::to_integer<unsigned>(byte) >> 4U];
                signature += kHex[std::to_integer<unsigned>(byte) & 15U];
            }
        }
        const models::qwen4_exp::PrefixCachePersistence loaded = instance->program->attach_prefix_cache_file(
            file, hybrid_cache_fingerprint(options, signature), options.startup_observer);
        restore = LoadSummary::PrefixCacheRestore{
            .attempted           = true,
            .restored            = loaded.ok,
            .message             = loaded.message,
            .blocks              = loaded.blocks,
            .snapshots           = loaded.snapshots,
            .bytes               = loaded.bytes,
            .seconds             = loaded.seconds,
            .saved_blocks        = loaded.saved_blocks,
            .saved_snapshots     = loaded.saved_snapshots,
            .required_host_bytes = loaded.required_host_bytes,
            .host_bytes          = loaded.host_bytes,
        };
    }
    program.complete();

    const auto& stats = instance->model->storage_stats();
    const auto& text  = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.prefill_signature    = prefill_signature;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul"};
    summary.context_cost         = std::move(context_cost.summary);
    summary.prefix_cache         = std::move(restore);
    summary.fast_path_configuration = models::qwen4_exp::describe(fast.configuration);
    for (const FastPath path : fast.paths) { summary.fast_paths.emplace_back(fast_path_name(path)); }
    summary.fast_paths_disabled = !options.fast_paths;
    // Tail skip makes the output depend on which experts the cache holds; the 8-bit routes' arithmetic
    // depends on the call's width.
    summary.request_invariant =
        fast.request_invariant() && options.expert_tail_skip == 0 && !options.a8_activations;

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = text.vocab_size;
    metadata.embedding_size = text.hidden_size;
    metadata.native_context = text.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = stats.device_capacity_bytes + stats.pinned_bytes;

    EngineOptions resolved = options;
    if (prefix_auto) { resolved.context_cache.host_capacity_bytes = prefix_bytes; }
    return {std::move(instance), std::move(summary), std::move(metadata), std::move(resolved)};
}

} // namespace infernix::runtime
