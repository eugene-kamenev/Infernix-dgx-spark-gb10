#include "models/w4_weights.h"

#include "infernix/ops/w4_g16.h"

#include <chrono>
#include <stdexcept>

namespace infernix::models {
namespace {

std::uint64_t align_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

Weight bf16_source_weight(const WeightParent& parent) {
    const WeightGeometry& g = parent.geometry;
    Weight w;
    w.payload          = parent.device;
    w.payload_bytes    = g.bytes;
    w.qtype            = QType::BF16;
    w.layout           = QuantLayout::Contiguous;
    w.ndim             = 2;
    w.n = w.shape[0] = w.padded_shape[0] = static_cast<std::int32_t>(g.shape[0]);
    w.k = w.shape[1] = w.padded_shape[1] = static_cast<std::int32_t>(g.shape[1]);
    w.qdata                              = parent.device;
    return w;
}

} // namespace

std::uint64_t w4_estimate(const WeightGeometry& g) {
    return align_256(weight_geometry(QType::NVFP4, QuantLayout::RowSplit, g.shape).bytes);
}

W4Weights quantize_w4_objects(const artifact::MaterializedArtifact& backing,
                              const std::vector<artifact::ObjectHandle>& objects, DeviceContext& device) {
    W4Weights out;
    if (objects.empty()) { return out; }
    const auto start = std::chrono::steady_clock::now();
    struct Item {
        const WeightParent* source;
        WeightGeometry geometry;
        std::uint64_t offset;
    };
    std::vector<Item> items;
    items.reserve(objects.size());
    std::uint64_t total = 0;
    for (const auto handle : objects) {
        const WeightParent& parent = backing.device_parent(handle);
        const WeightGeometry& g    = parent.geometry;
        if (g.format != QType::BF16 || g.layout != QuantLayout::Contiguous || g.shape.size() != 2 ||
            g.shape[1] % 16 != 0) {
            throw std::invalid_argument("w4 weights: a contiguous BF16 [n, k] matrix (k a multiple of 16) is needed");
        }
        const WeightGeometry qg = weight_geometry(QType::NVFP4, QuantLayout::RowSplit, g.shape);
        items.push_back({&parent, qg, total});
        total = align_256(total + qg.bytes);
        out.source_bytes += g.bytes;
    }
    out.storage = DeviceBuffer(total);
    for (const auto& item : items) {
        auto* at         = static_cast<std::byte*>(out.storage.p) + item.offset;
        void* codes      = at;
        void* row_scales = at + item.geometry.high_offset;
        void* scales     = at + item.geometry.scale_offset;
        (void)ops::quantize_bf16_w4_g16(bf16_source_weight(*item.source), codes, row_scales, scales, device.stream);
        auto parent      = std::make_unique<WeightParent>();
        parent->geometry = item.geometry;
        parent->data     = at;
        parent->device   = at;
        out.replaced.emplace(item.source, parent.get());
        out.parents.push_back(std::move(parent));
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void repoint(WeightView& view, const W4Weights& w4) {
    for (WeightRegion& part : view.parts) {
        if (const auto it = w4.replaced.find(part.parent); it != w4.replaced.end()) { part.parent = it->second; }
    }
}

} // namespace infernix::models
