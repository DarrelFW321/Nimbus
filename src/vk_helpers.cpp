#include "vk_helpers.h"

#include <algorithm>
#include <bit>
#include <fstream>

GpuImage createImage(VkDevice device, VmaAllocator allocator, const ImageDesc& desc) {
    GpuImage img;
    img.format = desc.format;
    img.extent = desc.extent;
    img.mips = desc.mips;

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = desc.extent;
    ci.mipLevels = desc.mips;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = desc.usage;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator, &ci, &ai, &img.image, &img.alloc, nullptr));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img.image;
    vi.viewType = desc.type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = desc.format;
    vi.subresourceRange = {desc.aspect, 0, desc.mips, 0, 1};
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &img.view));
    if (desc.mips > 1) {
        vi.subresourceRange.levelCount = 1;
        VK_CHECK(vkCreateImageView(device, &vi, nullptr, &img.mip0View));
    } else {
        img.mip0View = img.view;
    }

    if (desc.name) {
        debug::setName(device, VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(img.image), desc.name);
        debug::setName(device, VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<uint64_t>(img.view), desc.name);
    }
    return img;
}

void destroyImage(VkDevice device, VmaAllocator allocator, GpuImage& img) {
    if (img.mip0View != img.view) vkDestroyImageView(device, img.mip0View, nullptr);
    vkDestroyImageView(device, img.view, nullptr);
    if (img.image) vmaDestroyImage(allocator, img.image, img.alloc);
    img = {};
}

uint32_t mipCount(VkExtent3D e) {
    return std::bit_width(std::max({e.width, e.height, e.depth}));
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void generateMips(VkCommandBuffer cmd, const GpuImage& img) {
    auto half = [](uint32_t v) { return std::max(1u, v / 2); };
    VkExtent3D src = img.extent;
    for (uint32_t mip = 1; mip < img.mips; ++mip) {
        VkExtent3D dst{half(src.width), half(src.height), half(src.depth)};
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, 0, 1};
        blit.srcOffsets[1] = {int32_t(src.width), int32_t(src.height), int32_t(src.depth)};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        blit.dstOffsets[1] = {int32_t(dst.width), int32_t(dst.height), int32_t(dst.depth)};
        vkCmdBlitImage(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                       VK_FILTER_LINEAR);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        src = dst;
    }
}

VkShaderModule loadShaderModule(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader " + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * sizeof(uint32_t);
    ci.pCode = code.data();
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module));
    return module;
}

namespace debug {
namespace {
PFN_vkSetDebugUtilsObjectNameEXT pSetName = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT pBeginLabel = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT pEndLabel = nullptr;
}  // namespace

void init(VkInstance instance, bool available) {
    if (!available) return;
    pSetName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
    pBeginLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
    pEndLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
}

void setName(VkDevice device, VkObjectType type, uint64_t handle, const char* name) {
    if (!pSetName) return;
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    pSetName(device, &info);
}

void beginLabel(VkCommandBuffer cmd, const char* name) {
    if (!pBeginLabel) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    pBeginLabel(cmd, &label);
}

void endLabel(VkCommandBuffer cmd) {
    if (pEndLabel) pEndLabel(cmd);
}
}  // namespace debug
