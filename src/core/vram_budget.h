#pragma once

// Physical facts about one CUDA device's video memory: device-wide free memory, this process's OS
// budget (Windows WDDM, through DXGI) and whether the device drives a display. No policy: sizing
// decisions live with their owners (docs/maintainer/qwen3_8-flash-next-design.md §19.3.7).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace infernix {

enum class DisplayState : std::uint8_t {
    Headless, // no output attached and no display reported
    Attached, // an output is attached, or the driver reports an active display
    Unknown,  // neither source could tell; callers treat it as Attached
};

struct VramSnapshot {
    // cudaMemGetInfo (device-wide under WDDM); under WDDM with a DXGI budget, past the budget or with
    // several CUDA devices visible, the budget less this process's usage instead.
    std::size_t device_free  = 0;
    std::size_t device_total = 0;
    // A unified-memory GPU (cudaDevAttrIntegrated): device memory is system memory, so every
    // allocation of the process draws on one pool. Sizing must count the process's own host-side
    // allocations against it, and a shortfall of free memory can only come from other programs or
    // the kernel's reserve, never from a sysmem fallback that demotes this process's data.
    bool integrated          = false;
    // Integrated GPUs only: bytes of this same physical pool that an allocation can obtain even
    // though cudaMemGetInfo does not report them free, because the OS would reclaim them (file
    // cache and reclaimable slabs). Measured on GB10: cudaMemGetInfo's free equals MemFree while
    // tens of GiB of file cache sit in the LPDDR5x, and an allocation past it succeeds by evicting
    // that cache -- so sizing off device_free alone refuses memory the machine has. 0 elsewhere,
    // where the driver's number is already the whole dedicated pool.
    // device_free stays the driver's reading (it is what SpillGuard and the startup ledger report);
    // only sizing adds this term, and it never counts swap.
    std::uint64_t reclaimable = 0;
    // Windows WDDM: this process's local-memory budget and usage (DXGI QueryVideoMemoryInfo).
    bool has_budget            = false;
    std::uint64_t local_budget = 0;
    std::uint64_t local_usage  = 0;
    bool has_local_usage       = false;
    bool free_from_budget      = false;
    DisplayState display       = DisplayState::Unknown;
    int outputs                = -1; // DXGI outputs of the adapter; -1 unknown
    std::uint64_t budget_allowance = 0;
    // Windows WDDM: the adapter's shared system memory (DXGI_ADAPTER_DESC SharedSystemMemory) and
    // this process's use of it (NON_LOCAL CurrentUsage), 0 unknown; under WSL2 the same two facts
    // through DXCore (core/dxcore_adapter.h), native Linux 0. Every byte page-locked for the
    // device (cudaHostAlloc, cudaHostRegister) counts in the usage, and page-locking fails just
    // below the shared size (RTX 5090, 95.8 GiB of RAM: shared 76.61 GiB, locking failed at 76.07
    // GiB in 64 MiB steps, the same for mapped, unmapped and registered memory). Windows sets the
    // size, often half of RAM (80 % on that machine); the NON_LOCAL budget is not the limit.
    std::uint64_t shared_system  = 0;
    std::uint64_t nonlocal_usage = 0;
    std::uint64_t physical_used  = 0;
    std::uint64_t physical_free  = 0;
};

// One device's source. query() may be called from any thread on which the device is current.
class VramBudgetSource {
public:
    virtual ~VramBudgetSource()             = default;
    [[nodiscard]] virtual VramSnapshot query() = 0;
    [[nodiscard]] virtual void* change_event() { return nullptr; }

};

// The source for `device`: NVML where present (loaded lazily; its absence never fails, it only
// leaves fields unknown), else CUDA alone.
[[nodiscard]] std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device, bool past_budget = false);

// A device allocation step that the driver placed partly in system memory (the Windows WDDM sysmem
// fallback) lowers device free memory by less than it allocated. The bytes of a step of `bytes`
// missing from the drop in free memory beyond `tolerance`; 0: resident.
inline constexpr std::uint64_t kSpillToleranceBytes = 32ULL << 20;
[[nodiscard]] std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after,
                                            std::uint64_t bytes,
                                            std::uint64_t tolerance = kSpillToleranceBytes) noexcept;

// Checks device allocation steps against a source's free memory. A shortfall is confirmed by a
// second reading, since another program can change free memory during the step.
class SpillGuard {
public:
    explicit SpillGuard(VramBudgetSource& source) : source_(source) {}
    // Records free memory before a step.
    void begin() { before_ = source_.query(); }
    // After a step that allocated `bytes`: the bytes not placed in device memory (0: resident).
    [[nodiscard]] std::uint64_t end(std::uint64_t bytes);
    // The last step's readings, for its failure: free device memory before and after a step of
    // `bytes`, and where the source reads it (Windows, WSL2) this process's shared system memory use,
    // which a real sysmem fallback raises and a misread free-memory figure does not.
    [[nodiscard]] std::string readings(std::uint64_t bytes) const;

private:
    VramBudgetSource& source_;
    VramSnapshot before_;
    VramSnapshot after_;
};

namespace testing {
// Replaces every source opened afterwards with `fake` (tests of sizing and monitoring); an empty
// function restores the real sources.
void set_vram_budget_source(std::function<VramSnapshot()> fake);
} // namespace testing

} // namespace infernix
