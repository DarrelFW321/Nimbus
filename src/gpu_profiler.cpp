#include "gpu_profiler.h"

#include <algorithm>

#include "vk_helpers.h"

void GpuProfiler::init(VkDevice device, float timestampPeriodNs) {
    device_ = device;
    periodNs_ = timestampPeriodNs;
    for (FrameData& f : frames_) {
        VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = kMaxScopes * 2;
        VK_CHECK(vkCreateQueryPool(device, &ci, nullptr, &f.pool));
    }
}

void GpuProfiler::destroy() {
    for (FrameData& f : frames_) vkDestroyQueryPool(device_, f.pool, nullptr);
    frames_ = {};
}

void GpuProfiler::beginFrame(VkCommandBuffer cmd, uint32_t frame) {
    current_ = frame;
    FrameData& f = frames_[frame];
    f.names.clear();
    f.queryCount = 0;
    f.pending = true;
    openScopes_.clear();
    vkCmdResetQueryPool(cmd, f.pool, 0, kMaxScopes * 2);
}

void GpuProfiler::beginScope(VkCommandBuffer cmd, const char* name) {
    FrameData& f = frames_[current_];
    debug::beginLabel(cmd, name);
    if (f.names.size() >= kMaxScopes) return;
    uint32_t index = static_cast<uint32_t>(f.names.size());
    f.names.emplace_back(name);
    openScopes_.push_back(index);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.pool, index * 2);
}

void GpuProfiler::endScope(VkCommandBuffer cmd) {
    FrameData& f = frames_[current_];
    if (!openScopes_.empty()) {
        uint32_t index = openScopes_.back();
        openScopes_.pop_back();
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.pool, index * 2 + 1);
        f.queryCount = std::max(f.queryCount, index * 2 + 2);
    }
    debug::endLabel(cmd);
}

void GpuProfiler::resolve(uint32_t frame) {
    FrameData& f = frames_[frame];
    if (!f.pending || f.queryCount == 0) return;
    f.pending = false;

    uint64_t ts[kMaxScopes * 2];
    if (vkGetQueryPoolResults(device_, f.pool, 0, f.queryCount, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return;

    auto toMs = [&](uint64_t a, uint64_t b) { return float(double(b - a) * periodNs_ * 1e-6); };
    constexpr float kSmoothing = 0.05f;
    uint64_t first = UINT64_MAX, last = 0;
    for (size_t i = 0; i < f.names.size(); ++i) {
        uint64_t begin = ts[i * 2], end = ts[i * 2 + 1];
        first = std::min(first, begin);
        last = std::max(last, end);
        float ms = toMs(begin, end);

        auto it = std::find_if(scopes_.begin(), scopes_.end(), [&](const Scope& s) { return s.name == f.names[i]; });
        if (it == scopes_.end()) {
            scopes_.push_back({f.names[i], ms, ms});
        } else {
            it->lastMs = ms;
            it->smoothMs += (ms - it->smoothMs) * kSmoothing;
        }
    }
    lastOrder_ = f.names;
    frameMs_ = toMs(first, last);
    frameSmoothMs_ += (frameMs_ - frameSmoothMs_) * kSmoothing;
}
