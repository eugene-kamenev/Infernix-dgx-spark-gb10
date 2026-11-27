#pragma once

// Load-time 4-bit weights (LOSSY, EngineOptions::w4_weights): BF16 matrices converted to NVFP4 RowSplit
// (ops::quantize_bf16_w4_g16) in one device block, each object wrapped in its own WeightParent, its bound views
// repointed before freeing the transient BF16 copies. Shared by LoadPlan / Model.

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace infernix::models {

struct W4Weights {
    DeviceBuffer storage;
    std::vector<std::unique_ptr<WeightParent>> parents;
    std::unordered_map<const WeightParent*, const WeightParent*> replaced;
    std::uint64_t source_bytes = 0;
    double seconds             = 0;
};

[[nodiscard]] std::uint64_t w4_estimate(const WeightGeometry& geometry);

[[nodiscard]] W4Weights quantize_w4_objects(const artifact::MaterializedArtifact& backing,
                                            const std::vector<artifact::ObjectHandle>& objects,
                                            DeviceContext& device);

void repoint(WeightView& view, const W4Weights& w4);

} // namespace infernix::models
