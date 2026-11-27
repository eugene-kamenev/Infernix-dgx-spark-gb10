#include "core/cpu_topology.h"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace infernix {
namespace {

bool read_text(const std::string& path, std::string& out) {
    std::ifstream file(path);
    if (!file) { return false; }
    std::getline(file, out);
    return true;
}

bool parse_cpu_mask(const std::string& text, std::uint64_t& out) {
    out = 0;
    std::stringstream in(text);
    std::string part;
    while (std::getline(in, part, ',')) {
        if (part.empty() || part == "\n") { continue; }
        const auto dash = part.find('-');
        const int lo    = std::stoi(part.substr(0, dash));
        const int hi    = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
        if (lo < 0 || hi > 63 || lo > hi) { return false; }
        for (int c = lo; c <= hi; ++c) { out |= std::uint64_t{1} << c; }
    }
    return true;
}

} // namespace

std::vector<PhysicalCore> physical_cores() {
    std::vector<PhysicalCore> cores;
    std::uint64_t online = 0, performance = 0;
    std::string text;
    if (!read_text("/sys/devices/system/cpu/online", text) || !parse_cpu_mask(text, online)) { return cores; }

    bool hybrid = false;
    if (core::hybrid_cpu()) {
        hybrid = true;
        const auto perf_list = core::performance_cpus();
        for (int c : perf_list) {
            if (c >= 0 && c < 64) { performance |= (std::uint64_t{1} << c); }
        }
    } else if (read_text("/sys/devices/cpu_core/cpus", text) && parse_cpu_mask(text, performance)) {
        hybrid = true;
    }

    std::map<std::pair<int, int>, std::size_t> index; // (package, core id) -> cores[i]
    for (int cpu = 0; cpu < 64; ++cpu) {
        if (((online >> cpu) & 1U) == 0) { continue; }
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::string package, core;
        if (!read_text(base + "physical_package_id", package) || !read_text(base + "core_id", core)) { return {}; }
        const auto key = std::make_pair(std::stoi(package), std::stoi(core));
        const auto [it, added] = index.emplace(key, cores.size());
        if (added) {
            cores.push_back({0, static_cast<std::uint8_t>(hybrid && ((performance >> cpu) & 1U) != 0 ? 1 : 0)});
        }
        cores[it->second].cpus |= std::uint64_t{1} << cpu;
    }
    return cores;
}

std::uint32_t process_logical_cpus() noexcept {
#if defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process != 0) {
        return static_cast<std::uint32_t>(std::popcount(static_cast<std::uint64_t>(process)));
    }
    return static_cast<std::uint32_t>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) { return 0; }
    return static_cast<std::uint32_t>(CPU_COUNT(&set));
#else
    return 0;
#endif
}

int first_cpu(std::uint64_t cpus) noexcept { return cpus == 0 ? -1 : std::countr_zero(cpus); }

std::uint64_t set_current_thread_cpus(std::uint64_t cpus) noexcept {
    if (cpus == 0) { return 0; }
    cpu_set_t before;
    CPU_ZERO(&before);
    std::uint64_t previous = 0;
    if (pthread_getaffinity_np(pthread_self(), sizeof(before), &before) == 0) {
        for (int c = 0; c < 64; ++c) {
            if (CPU_ISSET(c, &before)) { previous |= std::uint64_t{1} << c; }
        }
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 64; ++c) {
        if (((cpus >> c) & 1U) != 0) { CPU_SET(c, &set); }
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0 ? previous : 0;
}

namespace core {
namespace {

// Reads one integer from a sysfs file; returns false when the file is absent or unreadable.
bool read_sysfs_long(std::string_view path, long long& out) {
    std::ifstream file{std::string(path)};
    if (!file) { return false; }
    std::string t;
    if (!(file >> t)) { return false; }
    const auto* begin = t.data();
    const auto* end   = t.data() + t.size();
    const auto result = std::from_chars(begin, end, out);
    return result.ec == std::errc{} && result.ptr == end;
}

// Expands a Linux cpulist ("0-19", "0-7,16-23") into the logical CPU ids it names.
std::vector<int> parse_cpu_list(std::string_view list) {
    std::vector<int> ids;
    std::size_t pos = 0;
    while (pos < list.size()) {
        const std::size_t comma = list.find(',', pos);
        const std::string_view item = list.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        const std::size_t dash    = item.find('-');
        long long low = -1, high = -1;
        if (dash == std::string_view::npos) {
            const auto r = std::from_chars(item.data(), item.data() + item.size(), low);
            if (r.ec == std::errc{}) { high = low; }
        } else {
            const std::string_view lo = item.substr(0, dash), hi = item.substr(dash + 1);
            const auto rl = std::from_chars(lo.data(), lo.data() + lo.size(), low);
            const auto rh = std::from_chars(hi.data(), hi.data() + hi.size(), high);
            if (rl.ec != std::errc{} || rh.ec != std::errc{}) { low = -1; }
        }
        if (low >= 0 && high >= low) {
            for (long long cpu = low; cpu <= high; ++cpu) { ids.push_back(static_cast<int>(cpu)); }
        }
        if (comma == std::string_view::npos) { break; }
        pos = comma + 1;
    }
    return ids;
}

// The online logical CPUs, from sysfs; empty when the file cannot be read.
std::vector<int> online_cpus() {
    std::ifstream file("/sys/devices/system/cpu/online");
    std::string t;
    if (!file || !(file >> t)) { return {}; }
    return parse_cpu_list(t);
}

// A capacity for each online CPU from one uniform source, so the values stay comparable: the
// normalized cpu_capacity (0..1024) if present for every CPU, else the cpufreq ceiling.
// Returns the per-CPU capacity in list order, or an empty vector if neither source covers all CPUs.
std::vector<long long> read_capacities(const std::vector<int>& cpus) {
    std::vector<long long> capacity(cpus.size(), 0);
    bool complete = true;
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (!read_sysfs_long("/sys/devices/system/cpu/cpu" + std::to_string(cpus[i]) + "/cpu_capacity", capacity[i])) {
            complete = false;
            break;
        }
    }
    if (complete) { return capacity; }
    complete = true;
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (!read_sysfs_long("/sys/devices/system/cpu/cpu" + std::to_string(cpus[i]) + "/cpufreq/cpuinfo_max_freq",
                             capacity[i])) {
            complete = false;
            break;
        }
    }
    return complete ? capacity : std::vector<long long>{};
}

// Performance-class ARM Cortex part numbers (midr_EL1 bits [15:4]), used only when neither sysfs
// capacity source is available. A core whose part is in this set counts as the big tier; a part not
// listed is treated as the little tier. Cortex-X series (X925 = 0xd85) and the performance Cortex-A
// cores; the observed Grace efficiency part (A725 = 0xd87) and Neoverse E cores are not listed.
bool arm_performance_part(long long midr) {
    const long long part = (midr >> 4) & 0xfff;
    static constexpr long long kBig[] = {0xd85, 0xd83, 0xd82, 0xd46, 0xd8d, 0xd81, 0xd47, 0xd48,
                                         0xd4d, 0xd41, 0xd42, 0xd0d, 0xd0e, 0xd0b, 0xd06};
    return std::any_of(std::begin(kBig), std::end(kBig), [part](long long p) { return p == part; });
}

} // namespace

bool hybrid_cpu() {
    const std::vector<int> cpus = online_cpus();
    if (cpus.size() < 2) { return false; }
    const std::vector<long long> capacity = read_capacities(cpus);
    if (!capacity.empty()) {
        const long long max_capacity = *std::max_element(capacity.begin(), capacity.end());
        const long long min_capacity = *std::min_element(capacity.begin(), capacity.end());
        // A real tier gap: the big tier is at least 20% above the small tier. Grace reads 997/718.
        return max_capacity >= min_capacity + min_capacity / 5;
    }
    // Fall back to the ARM part number for each CPU; hybrid when the set mixes big and small parts.
    bool big = false, little = false;
    for (const int cpu : cpus) {
        long long midr = 0;
        if (!read_sysfs_long("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/regs/identification/midr_el1",
                             midr)) {
            return false;
        }
        if (arm_performance_part(midr)) { big = true; } else { little = true; }
    }
    return big && little;
}

std::vector<int> performance_cpus() {
    if (!hybrid_cpu()) { return {}; }
    const std::vector<int> cpus = online_cpus();
    const std::vector<long long> capacity = read_capacities(cpus);
    std::vector<int> big;
    if (!capacity.empty()) {
        const long long max_capacity = *std::max_element(capacity.begin(), capacity.end());
        for (std::size_t i = 0; i < cpus.size(); ++i) {
            if (capacity[i] >= max_capacity - max_capacity / 10) { big.push_back(cpus[i]); }
        }
    } else {
        for (const int cpu : cpus) {
            long long midr = 0;
            if (!read_sysfs_long("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/regs/identification/midr_el1",
                                 midr)) {
                return {};
            }
            if (!arm_performance_part(midr)) {
                continue;
            }
            big.push_back(cpu);
        }
    }
    std::sort(big.begin(), big.end());
    return big;
}

} // namespace core

std::uint64_t slow_core_cpus() {
    static const std::uint64_t slow = [] {
        const std::vector<PhysicalCore> cores = physical_cores();
        if (cores.empty()) { return std::uint64_t{0}; }
        std::uint8_t lowest = 255, highest = 0;
        for (const PhysicalCore& core : cores) {
            lowest  = std::min(lowest, core.efficiency);
            highest = std::max(highest, core.efficiency);
        }
        std::uint64_t mask = 0;
        if (lowest != highest) {
            for (const PhysicalCore& core : cores) { mask |= core.efficiency == lowest ? core.cpus : 0; }
        }
        return mask;
    }();
    return slow;
}

} // namespace infernix
