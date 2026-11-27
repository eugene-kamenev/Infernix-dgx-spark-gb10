#pragma once

#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace infernix::runtime {

// Automatic capacity takes what `available_runtime_bytes` leaves after `automatic_headroom_bytes`;
// explicit capacity ignores the headroom.
[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes,
                                                       std::size_t automatic_headroom_bytes);

[[nodiscard]] std::size_t kv_available_bytes(bool automatic, std::size_t device_free,
                                             std::optional<std::uint64_t> budget_left,
                                             std::size_t automatic_headroom_bytes,
                                             std::size_t budget_margin_bytes) noexcept;

// The resolution of a plan whose runtime reservation grew beside `ordinary` (the same policy's
// resolution of the plan without the growth), when the growth keeps the KV capacity that
// resolution must keep: an explicit capacity whole, an automatic one down to every lane's full
// context (`lane_page_groups`) or to the ordinary capacity when that is smaller. Empty when the
// grown plan does not fit. Startup uses it to take an idle-wide-prefill width only where it fits.
[[nodiscard]] std::optional<KvCapacityResolution>
resolve_kv_capacity_beside(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                           std::size_t available_runtime_bytes,
                           std::size_t automatic_headroom_bytes,
                           const KvCapacityResolution& ordinary, std::uint64_t lane_page_groups);

[[nodiscard]] MemorySummary with_kv_capacity(MemorySummary summary, const KvCapacityResolution& resolution);

} // namespace infernix::runtime
