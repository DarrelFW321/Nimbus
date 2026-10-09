#pragma once
// Per-pass GPU timing with timestamp queries. Each scope also emits a debug-utils label so the
// same pass names show up in RenderDoc / Nsight captures.

#include <array>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

class GpuProfiler {
public:
    static constexpr uint32_t kMaxScopes = 32;
    static constexpr uint32_t kFrames = 2;

    struct Scope {
        std::string name;
        float lastMs = 0.0f;    // most recent resolved frame
        float smoothMs = 0.0f;  // exponential moving average for display
    };

    void init(VkDevice device, float timestampPeriodNs);
    void destroy();

    void beginFrame(VkCommandBuffer cmd, uint32_t frame);
    void beginScope(VkCommandBuffer cmd, const char* name);
    void endScope(VkCommandBuffer cmd);

    // Call after the frame's fence has signalled.
    void resolve(uint32_t frame);

    const std::vector<Scope>& scopes() const { return scopes_; }       // every scope ever seen
    const std::vector<std::string>& lastFrameOrder() const { return lastOrder_; }
    float frameMs() const { return frameMs_; }          // first scope start .. last scope end
    float frameSmoothMs() const { return frameSmoothMs_; }

private:
    struct FrameData {
        VkQueryPool pool = VK_NULL_HANDLE;
        std::vector<std::string> names;  // scope order this frame
        uint32_t queryCount = 0;
        bool pending = false;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    float periodNs_ = 1.0f;
    std::array<FrameData, kFrames> frames_{};
    uint32_t current_ = 0;
    std::vector<uint32_t> openScopes_;
    std::vector<Scope> scopes_;
    std::vector<std::string> lastOrder_;
    float frameMs_ = 0.0f, frameSmoothMs_ = 0.0f;
};
