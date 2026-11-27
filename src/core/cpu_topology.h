#pragma once

// The host CPU's physical cores and the logical CPUs of each (SMT siblings), with each core's
// efficiency class, so a placement can give threads one physical core each or keep a core apart.
// Logical CPUs are bits of a 64-bit mask (bit i: logical CPU i of the process's processor group);
// machines with more than 64 logical CPUs report no topology.
//
// Also exposes logical-CPU topology for a big.LITTLE (asymmetric) ARM/Grace CPU.
// The Grace host that runs the offloaded-expert CPU service is a 20-core ARMv9.2 part: ten
// Cortex-X925 performance cores and ten Cortex-A725 efficiency cores. A CPU expert round is
// throughput-bound on the workers that take it, so a worker parked on an efficiency core runs the
// same work ~40% slower than on a performance core. Pinning the team to the performance cores turns
// that into throughput; it must not be done on a symmetric host, where a fixed core subset would
// only remove parallelism.

#include <cstdint>
#include <vector>

namespace infernix {

struct PhysicalCore {
    std::uint64_t cpus      = 0; // the core's logical CPUs (two with SMT); the lowest bit is its first thread
    std::uint8_t efficiency = 0; // the OS's efficiency class: higher is a faster core (hybrid CPUs: P-cores)
};

// The physical cores in the OS's order (Linux: sysfs core ids and, on hybrid Intel/ARM CPUs,
// /sys/devices/cpu_core/cpus or normalized CPU capacities). Empty when the platform does not
// tell or the machine has more than 64 logical CPUs.
[[nodiscard]] std::vector<PhysicalCore> physical_cores();

// The logical CPUs this process may run on: its affinity (Windows: the process mask, else every
// active CPU; Linux: sched_getaffinity, so a taskset or a container's cpuset counts), not the
// machine's. 0 when unknown.
[[nodiscard]] std::uint32_t process_logical_cpus() noexcept;

// The lowest logical CPU of a non-empty mask.
[[nodiscard]] int first_cpu(std::uint64_t cpus) noexcept;

// Restricts the calling thread to the logical CPUs of `cpus` (0: no change) and returns its previous
// mask (0 when unknown or when the change failed).
std::uint64_t set_current_thread_cpus(std::uint64_t cpus) noexcept;

class ScopedThreadCpus {
public:
    explicit ScopedThreadCpus(std::uint64_t cpus) noexcept : previous_(set_current_thread_cpus(cpus)) {}
    ~ScopedThreadCpus() { restore(); }
    ScopedThreadCpus(const ScopedThreadCpus&)            = delete;
    ScopedThreadCpus& operator=(const ScopedThreadCpus&) = delete;
    void restore() noexcept {
        if (!restored_) { (void)set_current_thread_cpus(previous_); }
        restored_ = true;
    }

private:
    std::uint64_t previous_;
    bool restored_ = false;
};

[[nodiscard]] std::uint64_t slow_core_cpus();

namespace core {

// The performance-tier logical CPUs, sorted highest-capacity first; empty when the host is
// symmetric (all cores one tier) or the topology is unavailable. A homogeneous host therefore
// returns {} and the caller must not pin.
[[nodiscard]] std::vector<int> performance_cpus();

// True when the host exposes two or more performance tiers (a big.LITTLE / hybrid part).
[[nodiscard]] bool hybrid_cpu();

} // namespace core
} // namespace infernix
