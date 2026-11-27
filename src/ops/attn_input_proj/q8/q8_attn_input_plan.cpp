#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"

#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace infernix::ops::detail {

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8AttnInputScheduleId schedule;
};

// Retuned on GB10 (sm_121a, 48 SMs) via candidate sweeps + high-repeat edge sweeps. The HEAD
// tables inherited 5090 tuning: MmaR32C128 never wins on this device, and the narrow DFlash2 lanes
// (r16c64k128 / r32c32k128 / r32c64) are dominated by r32c64k128 (49..64) then r64c128 (65+).
// 2026-10-31 T8: the 128x128 tile takes 129 columns and above. Measured in isolated per-candidate
// processes (cold cache, warmup 3 / repeat 15, [9216,2048] parent; us, 128x128 against the 64x128
// incumbent): 129 137.2/143.2, 160 139.1/143.2, 192 141.1/145.2, 224 143.2/149.3, 288 204.6/206.7,
// 320 206.7/208.7, 416 247.6/272.2, 448 251.7/278.3, and 6-19% ahead from 512 in the full candidate
// sweep (4096: 2021.2 against 2165.4). Two widths stay ~3% behind - 256 (149.3 against 145.2) and
// 384 (212.0 against 204.8) - because the wider row tile halves the CTA count and lands on a
// fractional 48-SM wave there; freezing two single-width exceptions is not worth two more bands.
// The table is shared with the [17408,3072] MTP parent, whose 128x128 lane is oracle-tested at
// T = 129 and 300 but has no candidate-sweep route on this host (see the GB10 changelog).
constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, Q8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, Q8AttnInputScheduleId::MmaR64C128},
    {129, kAnyCols, Q8AttnInputScheduleId::MmaR128C128},
}};

// Companion alternates between MmaR64C128 and MmaR128C80 across token bands (tile-wave
// quantization on 48 SMs); both reproduce across independent sweeps with large gaps, so the
// alternation is captured rather than averaged away. SplitK now ends at 48 (it loses from 56+).
//
// 2026-10-31 T8: from 385 columns the 128x128 tile dominates both alternates. Isolated
// per-candidate processes (cold, warmup 3 / repeat 15, us, 128x128 against the routed tile):
// 384 137.0/161.6 and 512 171.8/216.9 against R128C80, 448 173.9/188.1, 608 216.9/235.4,
// 704 251.7/276.3 and 768 247.6/268.1 against R64C128, and 5-7% ahead out to 4096 in the full
// sweep (1322.8 against 1419.1). One width stays behind - 560, where R128C80 is 3% faster - which
// is not worth a single-column band.
constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 48, Q8AttnInputScheduleId::SplitKMmaDirect},
    {49, 132, Q8AttnInputScheduleId::MmaR64C128},
    {133, 164, Q8AttnInputScheduleId::MmaR128C80},
    {165, 192, Q8AttnInputScheduleId::MmaR64C96},
    {193, 256, Q8AttnInputScheduleId::MmaR64C128},
    {257, 320, Q8AttnInputScheduleId::MmaR128C80},
    {321, 384, Q8AttnInputScheduleId::MmaR64C128},
    {385, kAnyCols, Q8AttnInputScheduleId::MmaR128C128},
}};

// 2026-10-31 T8: the 128x128 tile takes 129 columns and above on every drafter width. Measured on
// the [6144,2560] Qwen3.8-Flash-Next drafter in isolated per-candidate processes (cold, warmup 3 /
// repeat 15, us, 128x128 against the 64x128 incumbent): 128 112.5/94.1 (loses, so 65..128 stays),
// 129 122.7/128.9, 192 124.7/133.2, 384 163.7/173.9, 512 206.7/223.1, 1024 393.0/432.0,
// 4096 1591.1/1757.0. The [6144,5120] drafter reproduces the same crossover in the candidate sweep
// (129 243.5/253.8, 4096 3225.4/3896.2).
constexpr std::array<RouteSpec, 4> kDFlash2Routes{{
    {1, 48, Q8AttnInputScheduleId::DFlash2SmallT},
    {49, 64, Q8AttnInputScheduleId::DFlash2MmaR32C64K128},
    {65, 128, Q8AttnInputScheduleId::DFlash2MmaR64C128},
    {129, kAnyCols, Q8AttnInputScheduleId::DFlash2MmaR128C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kTargetRoutes),
              "Q8 target attention input routes must be exact and closed");
static_assert(catalog_is_closed(kCompanionRoutes),
              "Q8 companion attention input routes must be exact and closed");
static_assert(catalog_is_closed(kDFlash2Routes),
              "Q8 DFlash2 attention input routes must be exact and closed");

bool is_companion_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool is_dflash2_shape(const Q8AttnInputProblem& problem) noexcept {
    // Qwen3.8-27B's drafter (H 5120), Qwen3.8-Flash-Next's (H 2560), Qwen3.5-122B-A10B's (H 3072)
    // and Inkling 2 Mini's (H 2304).
    return (problem.input_rows == 5120 || problem.input_rows == 2560 || problem.input_rows == 3072 ||
            problem.input_rows == 2304) &&
           problem.query_rows == 4096 &&
           problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == problem.input_rows;
}

// The four-output MTP parents: the Qwen3.6-35B-A3B's [9216, 2048] and the Qwen3.5-122B-A10B's
// [17408, 3072] (q8_attn_input_n17408.cu).
bool is_target_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.padded_k == 2048 && problem.query_rows == 4096 &&
           problem.kv_rows == 512 && problem.parent_rows == 9216;
}

bool is_n17408_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 3072 && problem.padded_k == 3072 && problem.query_rows == 8192 &&
           problem.kv_rows == 512 && problem.parent_rows == 17408;
}

bool supported_shape(const Q8AttnInputProblem& problem) noexcept {
    return is_target_shape(problem) || is_n17408_shape(problem) || is_companion_shape(problem) ||
           is_dflash2_shape(problem);
}

} // namespace

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.q8.decode.r8.direct.k2048";
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.q8.splitk.mma.r16.direct";
    case Q8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.q8.simt.r8.c4";
    case Q8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.q8.mma.r32.c64";
    case Q8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.q8.mma.r32.c128";
    case Q8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.q8.mma.r64.c64";
    case Q8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.q8.mma.r64.c96";
    case Q8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.q8.mma.r64.c128";
    case Q8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.q8.mma.r128.c64";
    case Q8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.q8.mma.r128.c80";
    case Q8AttnInputScheduleId::MmaR128C128:
        return "attn_input_proj.q8.mma.r128.c128";
    case Q8AttnInputScheduleId::DFlash2SmallT:
        return "attn_input_proj.q8.dflash2.small_t";
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
        return "attn_input_proj.q8.dflash2.mma.r16.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c32.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
        return "attn_input_proj.q8.dflash2.mma.r32.c64";
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
        return "attn_input_proj.q8.dflash2.mma.r64.c128";
    case Q8AttnInputScheduleId::DFlash2MmaR128C128:
        return "attn_input_proj.q8.dflash2.mma.r128.c128";
    case Q8AttnInputScheduleId::ColumnInvariant:
        return "attn_input_proj.q8.splitk.mma.column_invariant";
    }
    return "attn_input_proj.q8.unknown";
}

bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem,
                                           bool column_invariant) {
    if (!q8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q8 attention input: exact problem or column count is not admitted");
    }
    if (column_invariant) {
        if (is_dflash2_shape(problem) || is_companion_shape(problem)) {
            throw std::invalid_argument(
                "Q8 attention input: the problem registers no column-invariant route");
        }
        return {Q8AttnInputScheduleId::ColumnInvariant};
    }
    const auto resolve_from = [&](const auto& routes) -> Q8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("Q8 attention input: admitted problem has no covering route");
    };
    if (is_dflash2_shape(problem)) { return resolve_from(kDFlash2Routes); }
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    return resolve_from(kTargetRoutes);
}

namespace {

template <std::size_t N>
constexpr bool routes_contain(const std::array<RouteSpec, N>& routes,
                              Q8AttnInputScheduleId schedule) noexcept {
    for (const RouteSpec& route : routes) {
        if (route.schedule == schedule) { return true; }
    }
    return false;
}

} // namespace

bool q8_attn_input_candidate_is_legal(Q8AttnInputScheduleId schedule,
                                      const Q8AttnInputProblem& problem) noexcept {
    if (!supported_shape(problem) || problem.cols < 1) { return false; }
    const std::int32_t cols = problem.cols;
    // Each lane's kernel enforces its own column window; mirror it so the sweep never throws.
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        return cols == 1;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        if (is_dflash2_shape(problem)) { return false; }
        return cols >= 2 && cols <= (is_companion_shape(problem) ? 96 : 64);
    case Q8AttnInputScheduleId::DFlash2SmallT:
        return is_dflash2_shape(problem) && cols <= 48;
    default:
        break;
    }
    if (is_dflash2_shape(problem)) {
        return routes_contain(kDFlash2Routes, schedule) ||
               schedule == Q8AttnInputScheduleId::DFlash2MmaR128C128;
    }
    if (is_companion_shape(problem)) {
        return routes_contain(kCompanionRoutes, schedule) ||
               schedule == Q8AttnInputScheduleId::MmaR128C128;
    }
    // Target four-output shape: the routed schedules plus the SimtR8C4 lane, which the four-output
    // execute path supports but HEAD never routes to, and the 128x128 prefill tile under measurement.
    return routes_contain(kTargetRoutes, schedule) || schedule == Q8AttnInputScheduleId::SimtR8C4 ||
           schedule == Q8AttnInputScheduleId::MmaR128C128;
}

void q8_attn_input_execute_candidate(Q8AttnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    if (!q8_attn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("Q8 attention input: candidate schedule not legal for shape");
    }
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C128:
        q8_attn_input_mma_r128_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    default:
        throw std::logic_error("Q8 attention input: three-output schedule in four-output candidate");
    }
}

void q8_attn_input_execute_candidate(Q8AttnInputScheduleId schedule, const Tensor& x,
                                     const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                     cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    if (!q8_attn_input_candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("Q8 attention input: candidate schedule not legal for shape");
    }
    if (is_dflash2_shape(problem)) {
        switch (schedule) {
        case Q8AttnInputScheduleId::DFlash2SmallT:
            q8_dflash2_attn_input_small_t_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
            q8_dflash2_attn_input_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
            q8_dflash2_attn_input_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
            q8_dflash2_attn_input_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64:
            q8_dflash2_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR64C128:
            q8_dflash2_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR128C128:
            q8_dflash2_attn_input_mma_r128_c128_launch(x, weight, q, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: non-DFlash2 schedule in DFlash2 candidate");
        }
    }
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
        q8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C64:
        q8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C96:
        q8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C64:
        q8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C80:
        q8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C128:
        q8_attn_input_mma_r128_c128_launch(x, weight, q, k, v, stream);
        return;
    default:
        throw std::logic_error("Q8 attention input: schedule not legal for companion candidate");
    }
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream, bool column_invariant) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem, column_invariant);
    const bool n17408              = is_n17408_shape(problem);
    if ((!is_target_shape(problem) && !n17408) || resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact four-output problem");
    }
    if (n17408) {
        switch (plan.schedule) {
        case Q8AttnInputScheduleId::ColumnInvariant:
            q8_attn_input_n17408_column_invariant_launch(x, weight, q, gate, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DecodeR8Direct:
            q8_attn_input_n17408_decode_launch(x, weight, q, gate, k, v, stream);
            return;
        case Q8AttnInputScheduleId::SplitKMmaDirect:
            q8_attn_input_n17408_splitk_mma_launch(x, weight, q, gate, k, v, stream);
            return;
        case Q8AttnInputScheduleId::MmaR32C128:
            q8_attn_input_n17408_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
            return;
        case Q8AttnInputScheduleId::MmaR64C128:
            q8_attn_input_n17408_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
            return;
        case Q8AttnInputScheduleId::MmaR128C128:
            q8_attn_input_n17408_mma_r128_c128_launch(x, weight, q, gate, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: unregistered [17408,3072] schedule");
        }
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::ColumnInvariant:
        q8_attn_input_column_invariant_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C128:
        q8_attn_input_mma_r128_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
    case Q8AttnInputScheduleId::MmaR64C64:
    case Q8AttnInputScheduleId::MmaR64C96:
    case Q8AttnInputScheduleId::MmaR128C64:
    case Q8AttnInputScheduleId::MmaR128C80:
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR128C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: three-output schedule in four-output plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream, bool column_invariant) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem, column_invariant), x, weight, q,
                               gate, k, v, stream, column_invariant);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    const bool companion           = is_companion_shape(problem);
    const bool dflash2             = is_dflash2_shape(problem);
    if ((!companion && !dflash2) || resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact three-output problem");
    }
    if (dflash2) {
        switch (plan.schedule) {
        case Q8AttnInputScheduleId::DFlash2SmallT:
            q8_dflash2_attn_input_small_t_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
            q8_dflash2_attn_input_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
            q8_dflash2_attn_input_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
            q8_dflash2_attn_input_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64:
            q8_dflash2_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR64C128:
            q8_dflash2_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR128C128:
            q8_dflash2_attn_input_mma_r128_c128_launch(x, weight, q, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: non-DFlash2 schedule in DFlash2 plan");
        }
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
        q8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C64:
        q8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C96:
        q8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C64:
        q8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C80:
        q8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C128:
        q8_attn_input_mma_r128_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR128C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: DFlash2 schedule in companion plan");
    case Q8AttnInputScheduleId::ColumnInvariant:
        throw std::logic_error("Q8 attention input: four-output schedule in companion plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace infernix::ops::detail
