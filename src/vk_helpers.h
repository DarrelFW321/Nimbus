#pragma once
// Small Vulkan utilities: error checking, images, barriers, shader modules, debug labels.

#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define VK_CHECK(expr)                                                                              \
    do {                                                                                            \
        VkResult vkCheckResult_ = (expr);                                                           \
        if (vkCheckResult_ != VK_SUCCESS)                                                           \
            throw std::runtime_error(std::string(#expr " failed: VkResult ") + std::to_string(vkCheckResult_)); \
    } while (0)

struct GpuImage {
    VkImage       image = VK_NULL_HANDLE;
    VkImageView   view = VK_NULL_HANDLE;      // all mips (sampling)
    VkImageView   mip0View = VK_NULL_HANDLE;  // mip 0 only (storage writes); == view when mips == 1
    VmaAllocation alloc = nullptr;
    VkFormat      format = VK_FORMAT_UNDEFINED;
    VkExtent3D    extent{};
    uint32_t      mips = 1;
};

struct ImageDesc {
    VkImageType        type = VK_IMAGE_TYPE_2D;
    VkFormat           format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkExtent3D         extent{1, 1, 1};
    VkImageUsageFlags  usage = 0;
    uint32_t           mips = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    const char*        name = nullptr;
};

GpuImage createImage(VkDevice device, VmaAllocator allocator, const ImageDesc& desc);
void destroyImage(VkDevice device, VmaAllocator allocator, GpuImage& image);
uint32_t mipCount(VkExtent3D extent);

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

// Box-filters mip 0 down the chain with vkCmdBlitImage. Image must be in GENERAL layout.
void generateMips(VkCommandBuffer cmd, const GpuImage& image);

VkShaderModule loadShaderModule(VkDevice device, const std::string& spirvPath);

// VK_EXT_debug_utils labels and names (no-ops when the extension is unavailable).
namespace debug {
void init(VkInstance instance, bool available);
void setName(VkDevice device, VkObjectType type, uint64_t handle, const char* name);
void beginLabel(VkCommandBuffer cmd, const char* name);
void endLabel(VkCommandBuffer cmd);
}  // namespace debug
