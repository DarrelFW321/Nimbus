#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stb_image_write.h>

#include "blue_noise.h"
#include "shader_compiler.h"

namespace {

constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kTerrainVertsPerSide = 512;
constexpr float kTerrainSpacing = 80.0f;     // snap distance for the camera-centred grid
constexpr float kTerrainHalfExtent = 80000.0f;
constexpr float kTerrainTile = 40960.0f;     // metres covered by one heightmap repeat
constexpr float kShadowMapExtent = 40000.0f;
constexpr uint32_t kShadowMapSize = 512;
constexpr uint32_t kBlueNoiseSize = 64;

constexpr VkExtent2D kTransmittanceSize{256, 64};
constexpr VkExtent2D kMultiScatSize{32, 32};
constexpr VkExtent2D kSkyViewSize{192, 108};
constexpr uint32_t kAerialLutSize = 32;

constexpr VkPipelineStageFlags2 kComputeStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags2 kShaderStages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

template <typename T>
T unwrap(vkb::Result<T> result, const char* what) {
    if (!result) throw std::runtime_error(std::string(what) + ": " + result.error().message());
    return result.value();
}

uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Prefer ./shaders next to the working directory (relocatable builds), else the build tree.
std::string spirvPath(const char* name) {
    std::string local = std::string("shaders/") + name + ".spv";
    if (std::filesystem::exists(local)) return local;
    return std::string(SHADER_OUT_DIR) + "/" + name + ".spv";
}

// Bayer-ordered sub-pixel offsets so consecutive frames spread out across the N x N block.
glm::ivec2 bayerOffset(uint64_t frame, int n) {
    static constexpr int bayer4[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};
    static constexpr int bayer2[4] = {0, 2, 3, 1};
    int k = int(frame % uint64_t(n * n));
    for (int i = 0; i < n * n; ++i) {
        int rank = n == 4 ? bayer4[i] : n == 2 ? bayer2[i] : 0;
        if (rank == k) return {i % n, i / n};
    }
    return {0, 0};
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------
void Renderer::init(GLFWwindow* window, const RenderSettings& settings) {
    window_ = window;
    createDevice(window);
    createSwapchain(settings.vsync);
    createSamplers();
    createDescriptors();
    createFrames();
    createStaticResources();
    createTerrainIndexBuffer();
    createSizedResources(settings.upscaleFactor);
    writeDescriptors();
    profiler_.init(device_, physicalDevice_.properties.limits.timestampPeriod);

    currentWorkgroup_ = settings.workgroup;
    std::string log;
    if (!createPipelines(log)) throw std::runtime_error("Pipeline creation failed:\n" + log);
    runPrecompute();
    readBackHeightmap();
    initImGui();
}

void Renderer::shutdown() {
    if (device_.device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    destroyPipelines();
    profiler_.destroy();
    for (Frame& f : frames_) {
        vkDestroyCommandPool(device_, f.cmdPool, nullptr);
        vkDestroyFence(device_, f.inFlight, nullptr);
        vkDestroySemaphore(device_, f.imageAvailable, nullptr);
        vmaDestroyBuffer(allocator_, f.ubo, f.uboAlloc);
    }
    if (readbackBuffer_) vmaDestroyBuffer(allocator_, readbackBuffer_, readbackAlloc_);
    destroySizedResources();
    destroyStaticResources();
    for (VkSampler s : {linearClamp_, linearRepeat_, skySampler_, nearestClamp_}) vkDestroySampler(device_, s, nullptr);
    vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    vkDestroyDescriptorPool(device_, imguiPool_, nullptr);
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
    vkDestroyCommandPool(device_, immediatePool_, nullptr);
    destroySwapchain();

    vmaDestroyAllocator(allocator_);
    vkb::destroy_device(device_);
    vkb::destroy_surface(instance_, surface_);
    vkb::destroy_instance(instance_);
    device_ = {};
}

void Renderer::createDevice(GLFWwindow* window) {
    auto systemInfo = unwrap(vkb::SystemInfo::get_system_info(), "Vulkan system info");
    bool debugUtils = systemInfo.is_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    vkb::InstanceBuilder instanceBuilder;
    instanceBuilder.set_app_name("Nimbus").require_api_version(1, 3, 0);
    if (debugUtils) instanceBuilder.enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef ENABLE_VALIDATION
    instanceBuilder.request_validation_layers(true).use_default_debug_messenger();
#endif
    instance_ = unwrap(instanceBuilder.build(), "Instance creation");
    debug::init(instance_.instance, debugUtils);

    VK_CHECK(glfwCreateWindowSurface(instance_.instance, window, nullptr, &surface_));

    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.maintenance4 = VK_TRUE;  // LocalSizeId (spec-constant workgroup size)

    // Prefer the discrete GPU so hybrid laptops don't land on the iGPU.
    vkb::PhysicalDeviceSelector selector{instance_};
    physicalDevice_ = unwrap(selector.set_minimum_version(1, 3)
                                 .set_required_features_13(features13)
                                 .set_surface(surface_)
                                 .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                                 .select(),
                             "Physical device selection");

    device_ = unwrap(vkb::DeviceBuilder{physicalDevice_}.build(), "Device creation");
    queue_ = unwrap(device_.get_queue(vkb::QueueType::graphics), "Graphics queue");
    queueFamily_ = unwrap(device_.get_queue_index(vkb::QueueType::graphics), "Graphics queue index");

    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = physicalDevice_;
    allocInfo.device = device_;
    allocInfo.instance = instance_;
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&allocInfo, &allocator_));

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &immediatePool_));

    std::printf("GPU: %s\n", physicalDevice_.properties.deviceName);
}

template <typename F>
void Renderer::submitImmediate(F&& record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = immediatePool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    VK_CHECK(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, immediatePool_, 1, &cmd);
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------
void Renderer::createSwapchain(bool vsync) {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    vkb::SwapchainBuilder builder{device_, surface_};
    builder.set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .add_fallback_format({VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_extent(uint32_t(width), uint32_t(height))
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .set_old_swapchain(swapchain_);
    if (vsync) {
        builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR);
    } else {
        // Uncapped for profiling.
        builder.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
            .add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
            .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR);
    }
    vkb::Swapchain newSwapchain = unwrap(builder.build(), "Swapchain creation");

    destroySwapchain();
    swapchain_ = newSwapchain;
    swapImages_ = unwrap(swapchain_.get_images(), "Swapchain images");
    swapViews_ = unwrap(swapchain_.get_image_views(), "Swapchain image views");
    swapFormat_ = swapchain_.image_format;
    currentVsync_ = vsync;

    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    renderFinished_.resize(swapImages_.size());
    for (VkSemaphore& s : renderFinished_) VK_CHECK(vkCreateSemaphore(device_, &semInfo, nullptr, &s));
}

void Renderer::destroySwapchain() {
    for (VkSemaphore s : renderFinished_) vkDestroySemaphore(device_, s, nullptr);
    renderFinished_.clear();
    if (swapchain_.swapchain != VK_NULL_HANDLE) {
        swapchain_.destroy_image_views(swapViews_);
        vkb::destroy_swapchain(swapchain_);
    }
    swapViews_.clear();
    swapImages_.clear();
    swapchain_ = {};
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------
void Renderer::createSamplers() {
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.maxLod = VK_LOD_CLAMP_NONE;
    ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &linearClamp_));

    ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &linearRepeat_));

    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &skySampler_));

    ci.magFilter = ci.minFilter = VK_FILTER_NEAREST;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &nearestClamp_));
}

void Renderer::createDescriptors() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(B_Count);
    for (uint32_t b = 0; b < B_Count; ++b) {
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (b == B_Uniforms) type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b == B_TransmittanceImg || b == B_MultiScatImg || b == B_SkyViewImg || b == B_ShapeImg ||
                 b == B_DetailImg || b == B_WeatherImg || b == B_HeightImg || b == B_ShadowImg ||
                 b == B_CloudLowImg || b == B_CloudDepthImg || b == B_HistoryCurImg || b == B_HdrImg ||
                 b == B_AerialImg)
            type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[b] = {b, type, 1, VK_SHADER_STAGE_ALL, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = B_Count;
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &setLayout_));

    VkPushConstantRange push{VK_SHADER_STAGE_ALL, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));

    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight * B_Count},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kFramesInFlight * B_Count},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
    poolInfo.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
}

void Renderer::createFrames() {
    for (Frame& f : frames_) {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamily_;
        VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &f.cmdPool));

        VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdInfo.commandPool = f.cmdPool;
        cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &cmdInfo, &f.cmd));

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &f.inFlight));
        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device_, &semInfo, nullptr, &f.imageAvailable));

        VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo.size = sizeof(FrameUniforms);
        bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo allocResult{};
        VK_CHECK(vmaCreateBuffer(allocator_, &bufInfo, &ai, &f.ubo, &f.uboAlloc, &allocResult));
        f.uboMapped = allocResult.pMappedData;

        VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        setInfo.descriptorPool = descriptorPool_;
        setInfo.descriptorSetCount = 1;
        setInfo.pSetLayouts = &setLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &setInfo, &f.set));
    }
}

void Renderer::createStaticResources() {
    constexpr VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    constexpr VkImageUsageFlags mipped = rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    auto make2D = [&](VkFormat format, VkExtent2D e, VkImageUsageFlags usage, const char* name) {
        return createImage(device_, allocator_, {VK_IMAGE_TYPE_2D, format, {e.width, e.height, 1}, usage, 1,
                                                 VK_IMAGE_ASPECT_COLOR_BIT, name});
    };
    auto make3D = [&](uint32_t size, const char* name) {
        VkExtent3D e{size, size, size};
        return createImage(device_, allocator_, {VK_IMAGE_TYPE_3D, VK_FORMAT_R8G8B8A8_UNORM, e, mipped, mipCount(e),
                                                 VK_IMAGE_ASPECT_COLOR_BIT, name});
    };

    transmittanceLut_ = make2D(kHdrFormat, kTransmittanceSize, rw, "Transmittance LUT");
    multiScatLut_ = make2D(kHdrFormat, kMultiScatSize, rw, "Multi-scattering LUT");
    skyViewLut_ = make2D(kHdrFormat, kSkyViewSize, rw, "Sky-view LUT");
    aerialLut_ = createImage(device_, allocator_, {VK_IMAGE_TYPE_3D, kHdrFormat,
                                                   {kAerialLutSize, kAerialLutSize, kAerialLutSize}, rw, 1,
                                                   VK_IMAGE_ASPECT_COLOR_BIT, "Aerial perspective LUT"});
    shapeNoise_ = make3D(128, "Shape noise 128^3");
    detailNoise_ = make3D(32, "Detail noise 32^3");
    weatherMap_ = make2D(VK_FORMAT_R8G8B8A8_UNORM, {512, 512}, rw, "Weather map");
    heightmap_ = make2D(VK_FORMAT_R32_SFLOAT, {1024, 1024}, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "Heightmap");
    cloudShadow_ = make2D(VK_FORMAT_R32_SFLOAT, {kShadowMapSize, kShadowMapSize}, rw, "Cloud shadow map");
    blueNoise_ = make2D(VK_FORMAT_R8_UNORM, {kBlueNoiseSize, kBlueNoiseSize},
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "Blue noise");

    double t0 = glfwGetTime();
    blueNoiseData_ = generateBlueNoise(int(kBlueNoiseSize));
    std::printf("Blue noise (void-and-cluster %ux%u): %.0f ms\n", kBlueNoiseSize, kBlueNoiseSize,
                (glfwGetTime() - t0) * 1000.0);
}

void Renderer::destroyStaticResources() {
    for (GpuImage* img : {&transmittanceLut_, &multiScatLut_, &skyViewLut_, &aerialLut_, &shapeNoise_,
                          &detailNoise_, &weatherMap_, &heightmap_, &cloudShadow_, &blueNoise_})
        destroyImage(device_, allocator_, *img);
    if (terrainIndices_) vmaDestroyBuffer(allocator_, terrainIndices_, terrainIndicesAlloc_);
    terrainIndices_ = VK_NULL_HANDLE;
}

void Renderer::createSizedResources(int upscaleFactor) {
    VkExtent2D full = swapchain_.extent;
    lowExtent_ = {divUp(full.width, uint32_t(upscaleFactor)), divUp(full.height, uint32_t(upscaleFactor))};
    currentUpscale_ = upscaleFactor;

    constexpr VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    auto make = [&](VkFormat format, VkExtent2D e, VkImageUsageFlags usage, VkImageAspectFlags aspect, const char* name) {
        return createImage(device_, allocator_, {VK_IMAGE_TYPE_2D, format, {e.width, e.height, 1}, usage, 1, aspect, name});
    };
    sceneColor_ = make(kHdrFormat, full, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT, "Scene color");
    sceneDepth_ = make(kDepthFormat, full, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT, "Scene depth (reverse-Z)");
    cloudLow_ = make(kHdrFormat, lowExtent_, rw, VK_IMAGE_ASPECT_COLOR_BIT, "Clouds (low-res)");
    cloudDepth_ = make(VK_FORMAT_R32_SFLOAT, lowExtent_, rw, VK_IMAGE_ASPECT_COLOR_BIT, "Cloud depth (low-res)");
    history_[0] = make(kHdrFormat, full, rw, VK_IMAGE_ASPECT_COLOR_BIT, "Cloud history A");
    history_[1] = make(kHdrFormat, full, rw, VK_IMAGE_ASPECT_COLOR_BIT, "Cloud history B");
    hdr_ = make(kHdrFormat, full, rw, VK_IMAGE_ASPECT_COLOR_BIT, "HDR composite");

    // Resting layouts: everything GENERAL, depth read-only between frames.
    submitImmediate([&](VkCommandBuffer cmd) {
        for (GpuImage* img : {&sceneColor_, &cloudLow_, &cloudDepth_, &history_[0], &history_[1], &hdr_})
            imageBarrier(cmd, img->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        imageBarrier(cmd, sceneDepth_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    });
    historyValid_ = false;
}

void Renderer::destroySizedResources() {
    for (GpuImage* img : {&sceneColor_, &sceneDepth_, &cloudLow_, &cloudDepth_, &history_[0], &history_[1], &hdr_})
        destroyImage(device_, allocator_, *img);
}

void Renderer::writeDescriptors() {
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        std::array<VkDescriptorImageInfo, B_Count> images{};
        VkDescriptorBufferInfo uboInfo{frames_[i].ubo, 0, sizeof(FrameUniforms)};
        auto tex = [&](Binding b, const GpuImage& img, VkSampler sampler,
                       VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) { images[b] = {sampler, img.view, layout}; };
        auto storage = [&](Binding b, const GpuImage& img) {
            images[b] = {VK_NULL_HANDLE, img.mip0View, VK_IMAGE_LAYOUT_GENERAL};
        };

        tex(B_TransmittanceTex, transmittanceLut_, linearClamp_);  storage(B_TransmittanceImg, transmittanceLut_);
        tex(B_MultiScatTex, multiScatLut_, linearClamp_);          storage(B_MultiScatImg, multiScatLut_);
        tex(B_SkyViewTex, skyViewLut_, skySampler_);               storage(B_SkyViewImg, skyViewLut_);
        tex(B_ShapeTex, shapeNoise_, linearRepeat_);               storage(B_ShapeImg, shapeNoise_);
        tex(B_DetailTex, detailNoise_, linearRepeat_);             storage(B_DetailImg, detailNoise_);
        tex(B_WeatherTex, weatherMap_, linearRepeat_);             storage(B_WeatherImg, weatherMap_);
        tex(B_HeightTex, heightmap_, linearRepeat_);               storage(B_HeightImg, heightmap_);
        tex(B_ShadowTex, cloudShadow_, linearClamp_);              storage(B_ShadowImg, cloudShadow_);
        tex(B_BlueNoiseTex, blueNoise_, nearestClamp_);
        tex(B_SceneColorTex, sceneColor_, linearClamp_);
        tex(B_SceneDepthTex, sceneDepth_, nearestClamp_, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        tex(B_CloudLowTex, cloudLow_, linearClamp_);               storage(B_CloudLowImg, cloudLow_);
        tex(B_CloudDepthTex, cloudDepth_, nearestClamp_);          storage(B_CloudDepthImg, cloudDepth_);
        // Ping-pong: frame i reads the history written by the other frame-in-flight.
        tex(B_HistoryPrevTex, history_[1 - i], linearClamp_);      storage(B_HistoryCurImg, history_[i]);
        tex(B_HdrTex, hdr_, linearClamp_);                         storage(B_HdrImg, hdr_);
        tex(B_AerialTex, aerialLut_, linearClamp_);                storage(B_AerialImg, aerialLut_);

        std::array<VkWriteDescriptorSet, B_Count> writes{};
        for (uint32_t b = 0; b < B_Count; ++b) {
            VkWriteDescriptorSet& w = writes[b];
            w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = frames_[i].set;
            w.dstBinding = b;
            w.descriptorCount = 1;
            if (b == B_Uniforms) {
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w.pBufferInfo = &uboInfo;
            } else {
                w.descriptorType = images[b].sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                     : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w.pImageInfo = &images[b];
            }
        }
        vkUpdateDescriptorSets(device_, B_Count, writes.data(), 0, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------
void Renderer::createCloudPipeline() {
    glm::uvec2 wg = kWorkgroupSizes[currentWorkgroup_];
    VkSpecializationMapEntry entries[2] = {{0, 0, 4}, {1, 4, 4}};
    VkSpecializationInfo spec{2, entries, sizeof(wg), &wg};

    VkShaderModule module = loadShaderModule(device_, spirvPath("clouds.comp"));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = &spec;
    ci.layout = pipelineLayout_;
    VkResult r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &computePipelines_[CP_Clouds]);
    vkDestroyShaderModule(device_, module, nullptr);
    VK_CHECK(r);
}

bool Renderer::createPipelines(std::string& log) {
    static constexpr const char* computeShaders[CP_Count] = {
        "transmittance_lut.comp", "multiscatter_lut.comp", "skyview_lut.comp", "noise_shape.comp",
        "noise_detail.comp", "weather.comp", "heightmap.comp", "cloud_shadow.comp", "clouds.comp", "reconstruct.comp",
        "aerial_lut.comp"};
    try {
        for (uint32_t i = 0; i < CP_Count; ++i) {
            if (i == CP_Clouds) {
                createCloudPipeline();
                continue;
            }
            VkShaderModule module = loadShaderModule(device_, spirvPath(computeShaders[i]));
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = module;
            ci.stage.pName = "main";
            ci.layout = pipelineLayout_;
            VkResult r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &computePipelines_[i]);
            vkDestroyShaderModule(device_, module, nullptr);
            VK_CHECK(r);
        }

        // Shared fixed-function state for the two graphics pipelines.
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &blendAttachment;
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamicStates;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

        auto buildGraphics = [&](const char* vsName, const char* fsName, VkFormat colorFormat, bool useDepth) {
            VkShaderModule vs = loadShaderModule(device_, spirvPath(vsName));
            VkShaderModule fs = loadShaderModule(device_, spirvPath(fsName));
            bool hlsl = std::string(vsName).find(".vs") != std::string::npos;
            VkPipelineShaderStageCreateInfo stages[2] = {
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs,
                 hlsl ? "VSMain" : "main", nullptr},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs,
                 hlsl ? "PSMain" : "main", nullptr},
            };
            depth.depthTestEnable = useDepth;
            depth.depthWriteEnable = useDepth;
            depth.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;  // reverse-Z
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &colorFormat;
            rendering.depthAttachmentFormat = useDepth ? kDepthFormat : VK_FORMAT_UNDEFINED;

            VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpi.pNext = &rendering;
            gpi.stageCount = 2;
            gpi.pStages = stages;
            gpi.pVertexInputState = &vertexInput;
            gpi.pInputAssemblyState = &inputAssembly;
            gpi.pViewportState = &viewport;
            gpi.pRasterizationState = &raster;
            gpi.pMultisampleState = &multisample;
            gpi.pDepthStencilState = &depth;
            gpi.pColorBlendState = &blend;
            gpi.pDynamicState = &dynamic;
            gpi.layout = pipelineLayout_;
            VkPipeline pipeline;
            VkResult r = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline);
            vkDestroyShaderModule(device_, vs, nullptr);
            vkDestroyShaderModule(device_, fs, nullptr);
            VK_CHECK(r);
            return pipeline;
        };
        terrainPipeline_ = buildGraphics("terrain.vert", "terrain.frag", kHdrFormat, true);
        tonemapPipeline_ = buildGraphics("tonemap.vs", "tonemap.ps", swapFormat_, false);
    } catch (const std::exception& e) {
        log = e.what();
        pipelinesValid_ = false;
        return false;
    }
    pipelinesValid_ = true;
    return true;
}

void Renderer::destroyPipelines() {
    for (VkPipeline& p : computePipelines_) {
        vkDestroyPipeline(device_, p, nullptr);
        p = VK_NULL_HANDLE;
    }
    vkDestroyPipeline(device_, terrainPipeline_, nullptr);
    vkDestroyPipeline(device_, tonemapPipeline_, nullptr);
    terrainPipeline_ = tonemapPipeline_ = VK_NULL_HANDLE;
    pipelinesValid_ = false;
}

bool Renderer::reloadShaders(std::string& log) {
    if (!compileAllShaders(log)) return false;
    vkDeviceWaitIdle(device_);
    destroyPipelines();
    if (!createPipelines(log)) return false;
    runPrecompute();
    readBackHeightmap();
    historyValid_ = false;
    log = "Shaders reloaded.";
    return true;
}

// One-time generation of LUTs, noise volumes, weather map, heightmap and blue noise.
void Renderer::runPrecompute() {
    VkBuffer staging;
    VmaAllocation stagingAlloc;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = blueNoiseData_.size();
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &staging, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, blueNoiseData_.data(), blueNoiseData_.size());

    double t0 = glfwGetTime();
    submitImmediate([&](VkCommandBuffer cmd) {
        debug::beginLabel(cmd, "Precompute");
        for (GpuImage* img : {&transmittanceLut_, &multiScatLut_, &skyViewLut_, &aerialLut_, &shapeNoise_,
                              &detailNoise_, &weatherMap_, &heightmap_, &cloudShadow_, &blueNoise_})
            imageBarrier(cmd, img->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kBlueNoiseSize, kBlueNoiseSize, 1};
        vkCmdCopyBufferToImage(cmd, staging, blueNoise_.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &frames_[0].set, 0, nullptr);
        auto dispatch = [&](ComputePass pass, uint32_t x, uint32_t y, uint32_t z) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelines_[pass]);
            vkCmdDispatch(cmd, x, y, z);
        };
        auto computeToCompute = [&] {
            memoryBarrier(cmd, kComputeStage, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, kComputeStage,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        };

        dispatch(CP_Transmittance, divUp(kTransmittanceSize.width, 8), divUp(kTransmittanceSize.height, 8), 1);
        computeToCompute();
        dispatch(CP_MultiScatter, divUp(kMultiScatSize.width, 8), divUp(kMultiScatSize.height, 8), 1);
        dispatch(CP_NoiseShape, 128 / 4, 128 / 4, 128 / 4);
        dispatch(CP_NoiseDetail, 32 / 4, 32 / 4, 32 / 4);
        dispatch(CP_Weather, 512 / 8, 512 / 8, 1);
        dispatch(CP_Heightmap, 1024 / 8, 1024 / 8, 1);

        memoryBarrier(cmd, kComputeStage, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        generateMips(cmd, shapeNoise_);
        generateMips(cmd, detailNoise_);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        debug::endLabel(cmd);
    });
    std::printf("GPU precompute (LUTs, noise volumes, weather, heightmap): %.0f ms\n", (glfwGetTime() - t0) * 1000.0);
    vmaDestroyBuffer(allocator_, staging, stagingAlloc);
}

void Renderer::createTerrainIndexBuffer() {
    constexpr uint32_t n = kTerrainVertsPerSide;
    std::vector<uint32_t> indices;
    indices.reserve(size_t(n - 1) * (n - 1) * 6);
    for (uint32_t z = 0; z + 1 < n; ++z)
        for (uint32_t x = 0; x + 1 < n; ++x) {
            uint32_t i = z * n + x;
            indices.insert(indices.end(), {i, i + n, i + 1, i + 1, i + n, i + n + 1});
        }
    terrainIndexCount_ = uint32_t(indices.size());
    VkDeviceSize bytes = indices.size() * sizeof(uint32_t);

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &terrainIndices_, &terrainIndicesAlloc_, nullptr));

    VkBuffer staging;
    VmaAllocation stagingAlloc;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &staging, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, indices.data(), bytes);
    submitImmediate([&](VkCommandBuffer cmd) {
        VkBufferCopy copy{0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging, terrainIndices_, 1, &copy);
    });
    vmaDestroyBuffer(allocator_, staging, stagingAlloc);
}

void Renderer::readBackHeightmap() {
    const VkExtent3D e = heightmap_.extent;
    VkBuffer buffer;
    VmaAllocation alloc;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(e.width) * e.height * sizeof(float);
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &buffer, &alloc, &info));

    submitImmediate([&](VkCommandBuffer cmd) {
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = e;
        vkCmdCopyImageToBuffer(cmd, heightmap_.image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    });
    VK_CHECK(vmaInvalidateAllocation(allocator_, alloc, 0, VK_WHOLE_SIZE));
    const float* src = static_cast<const float*>(info.pMappedData);
    heightData_.assign(src, src + size_t(e.width) * e.height);
    vmaDestroyBuffer(allocator_, buffer, alloc);
}

float Renderer::terrainHeight(float x, float z, float heightScale) const {
    if (heightData_.empty()) return 0.0f;
    const int size = int(heightmap_.extent.width);
    // Texel centres at (i + 0.5) / size, REPEAT addressing - same as the GPU sampler.
    float fx = x / kTerrainTile * float(size) - 0.5f;
    float fz = z / kTerrainTile * float(size) - 0.5f;
    int x0 = int(std::floor(fx)), z0 = int(std::floor(fz));
    float tx = fx - float(x0), tz = fz - float(z0);
    auto at = [&](int ix, int iz) {
        ix = ((ix % size) + size) % size;
        iz = ((iz % size) + size) % size;
        return heightData_[size_t(iz) * size + ix];
    };
    float h0 = at(x0, z0) + (at(x0 + 1, z0) - at(x0, z0)) * tx;
    float h1 = at(x0, z0 + 1) + (at(x0 + 1, z0 + 1) - at(x0, z0 + 1)) * tx;
    return (h0 + (h1 - h0) * tz) * heightScale;
}

// ---------------------------------------------------------------------------
// ImGui
// ---------------------------------------------------------------------------
void Renderer::initImGui() {
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &size;
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &imguiPool_));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window_, true);

    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = instance_;
    info.PhysicalDevice = physicalDevice_;
    info.Device = device_;
    info.QueueFamily = queueFamily_;
    info.Queue = queue_;
    info.DescriptorPool = imguiPool_;
    info.MinImageCount = 2;
    info.ImageCount = static_cast<uint32_t>(swapImages_.size());
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapFormat_;
    ImGui_ImplVulkan_Init(&info);
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------
void Renderer::handleSettingsChanges(const RenderSettings& settings) {
    if (swapchainDirty_ || settings.vsync != currentVsync_) {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {  // minimised
            glfwWaitEvents();
            glfwGetFramebufferSize(window_, &width, &height);
        }
        vkDeviceWaitIdle(device_);
        createSwapchain(settings.vsync);
        destroySizedResources();
        createSizedResources(settings.upscaleFactor);
        writeDescriptors();
        swapchainDirty_ = false;
    } else if (settings.upscaleFactor != currentUpscale_) {
        vkDeviceWaitIdle(device_);
        destroySizedResources();
        createSizedResources(settings.upscaleFactor);
        writeDescriptors();
    }
    if (settings.workgroup != currentWorkgroup_ && pipelinesValid_) {
        vkDeviceWaitIdle(device_);
        vkDestroyPipeline(device_, computePipelines_[CP_Clouds], nullptr);
        currentWorkgroup_ = settings.workgroup;
        createCloudPipeline();
    }
}

FrameUniforms Renderer::buildUniforms(const CameraState& camera, const RenderSettings& s, float time) const {
    glm::vec3 sunDir{std::cos(s.sunElevation) * std::sin(s.sunAzimuth), std::sin(s.sunElevation),
                     -std::cos(s.sunElevation) * std::cos(s.sunAzimuth)};
    glm::vec2 camXZ{camera.position.x, camera.position.z};
    glm::vec2 gridOrigin = glm::floor(camXZ / kTerrainSpacing) * kTerrainSpacing;
    float shadowTexel = kShadowMapExtent / float(kShadowMapSize);
    glm::vec2 shadowCenter = glm::floor(camXZ / shadowTexel) * shadowTexel;
    glm::ivec2 offset = bayerOffset(frameCounter_, s.upscaleFactor);

    FrameUniforms u{};
    u.viewProj = camera.viewProj;
    u.invViewProj = glm::inverse(camera.viewProj);
    u.prevViewProj = hasPrevViewProj_ ? prevViewProj_ : camera.viewProj;
    u.cameraPos = glm::vec4(camera.position, time);
    u.sunDir = glm::vec4(glm::normalize(sunDir), s.sunIlluminance);
    u.cloudLayer = {s.layerBottom, s.layerTop, s.coverage, s.density};
    u.cloudNoise = {1.0f / s.shapeTile, 1.0f / s.detailTile, 1.0f / s.weatherTile, s.detailErosion};
    u.cloudLight = {s.ambient, s.powder, s.windSpeed, s.temporalBlend};
    u.terrain = {kTerrainTile, s.terrainHeight, kTerrainSpacing, float(kTerrainVertsPerSide)};
    u.terrainOrigin = {gridOrigin.x, gridOrigin.y, kShadowMapExtent, kTerrainHalfExtent};
    u.shadowOrigin = {shadowCenter.x, shadowCenter.y, s.fadeDistance, 0.0f};
    u.march = {s.primarySteps, s.lightSteps, int(frameCounter_ & 0x7fffffff), s.flags};
    u.recon = {s.upscaleFactor, offset.x, offset.y, historyValid_ ? 1 : 0};
    u.view = {s.debugView, int(swapchain_.extent.width), int(swapchain_.extent.height), s.terrain ? 1 : 0};
    return u;
}

void Renderer::drawFrame(const CameraState& camera, const RenderSettings& settings, float time) {
    handleSettingsChanges(settings);

    Frame& frame = frames_[frameIndex_];
    VK_CHECK(vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
    profiler_.resolve(frameIndex_);

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, frame.imageAvailable,
                                             VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainDirty_ = true;
        return;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) VK_CHECK(acquire);
    VK_CHECK(vkResetFences(device_, 1, &frame.inFlight));

    FrameUniforms uniforms = buildUniforms(camera, settings, time);
    std::memcpy(frame.uboMapped, &uniforms, sizeof(uniforms));

    bool capture = !screenshotPath_.empty();
    if (capture) {
        VkExtent2D e = swapchain_.extent;
        if (!readbackBuffer_ || readbackExtent_.width != e.width || readbackExtent_.height != e.height) {
            if (readbackBuffer_) vmaDestroyBuffer(allocator_, readbackBuffer_, readbackAlloc_);
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = VkDeviceSize(e.width) * e.height * 4;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VK_CHECK(vmaCreateBuffer(allocator_, &bi, &ai, &readbackBuffer_, &readbackAlloc_, nullptr));
            readbackExtent_ = e;
        }
    }

    VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));
    recordFrame(frame, imageIndex, settings);

    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = frame.cmd;
    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInfo.semaphore = frame.imageAvailable;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalInfo.semaphore = renderFinished_[imageIndex];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitInfo;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalInfo;
    VK_CHECK(vkQueueSubmit2(queue_, 1, &submit, frame.inFlight));

    if (capture) {
        VK_CHECK(vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
        saveScreenshot();
    }

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished_[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_.swapchain;
    present.pImageIndices = &imageIndex;
    VkResult presentResult = vkQueuePresentKHR(queue_, &present);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        swapchainDirty_ = true;
    else
        VK_CHECK(presentResult);

    prevViewProj_ = camera.viewProj;
    hasPrevViewProj_ = true;
    historyValid_ = true;
    ++frameCounter_;
    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
}

void Renderer::recordFrame(Frame& frame, uint32_t imageIndex, const RenderSettings& settings) {
    VkCommandBuffer cmd = frame.cmd;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    profiler_.beginFrame(cmd, frameIndex_);

    const VkExtent2D full = swapchain_.extent;
    VkViewport viewport{0.0f, 0.0f, float(full.width), float(full.height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, full};

    // Cross-frame hazards (history ping-pong, shared low-res targets) are covered by one full barrier.
    memoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

    if (pipelinesValid_) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);
        auto dispatch = [&](ComputePass pass, uint32_t x, uint32_t y) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelines_[pass]);
            vkCmdDispatch(cmd, x, y, 1);
        };
        auto computeWriteToRead = [&](VkPipelineStageFlags2 dstStages) {
            memoryBarrier(cmd, kComputeStage, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, dstStages,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        };

        profiler_.beginScope(cmd, "Sky-view LUT");
        dispatch(CP_SkyView, divUp(kSkyViewSize.width, 8), divUp(kSkyViewSize.height, 8));
        profiler_.endScope(cmd);
        computeWriteToRead(kShaderStages);

        if (settings.flags & FLAG_AERIAL_LUT) {
            profiler_.beginScope(cmd, "Aerial perspective LUT");
            dispatch(CP_AerialPerspective, kAerialLutSize / 8, kAerialLutSize / 8);
            profiler_.endScope(cmd);
        }

        profiler_.beginScope(cmd, "Cloud shadow map");
        dispatch(CP_CloudShadow, kShadowMapSize / 8, kShadowMapSize / 8);
        profiler_.endScope(cmd);
        computeWriteToRead(kShaderStages);

        // --- Terrain (raster, reverse-Z) -----------------------------------------------------
        profiler_.beginScope(cmd, "Terrain");
        imageBarrier(cmd, sceneColor_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     kComputeStage, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        imageBarrier(cmd, sceneDepth_.image, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, kComputeStage, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT);

        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = sceneColor_.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = sceneDepth_.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0};  // reverse-Z: 0 = infinitely far
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, full};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        rendering.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &rendering);
        if (settings.terrain) {
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline_);
            if (settings.flags & FLAG_INDEXED_TERRAIN) {
                vkCmdBindIndexBuffer(cmd, terrainIndices_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, terrainIndexCount_, 1, 0, 0, 0);
            } else {
                constexpr uint32_t quads = (kTerrainVertsPerSide - 1) * (kTerrainVertsPerSide - 1);
                vkCmdDraw(cmd, quads * 6, 1, 0, 0);
            }
        }
        vkCmdEndRendering(cmd);

        imageBarrier(cmd, sceneColor_.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     kComputeStage, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        imageBarrier(cmd, sceneDepth_.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, kComputeStage, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT);
        profiler_.endScope(cmd);

        // --- Clouds ------------------------------------------------------------------------
        profiler_.beginScope(cmd, "Clouds (ray march)");
        glm::uvec2 wg = kWorkgroupSizes[currentWorkgroup_];
        dispatch(CP_Clouds, divUp(lowExtent_.width, wg.x), divUp(lowExtent_.height, wg.y));
        profiler_.endScope(cmd);
        computeWriteToRead(kComputeStage);

        profiler_.beginScope(cmd, "Temporal reconstruct");
        dispatch(CP_Reconstruct, divUp(full.width, 8), divUp(full.height, 8));
        profiler_.endScope(cmd);
        computeWriteToRead(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    }

    // --- Tonemap + UI into the swapchain ---------------------------------------------------
    profiler_.beginScope(cmd, "Tonemap + UI");
    imageBarrier(cmd, swapImages_[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo swapColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    swapColor.imageView = swapViews_[imageIndex];
    swapColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    swapColor.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    swapColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo swapRendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    swapRendering.renderArea = {{0, 0}, full};
    swapRendering.layerCount = 1;
    swapRendering.colorAttachmentCount = 1;
    swapRendering.pColorAttachments = &swapColor;
    vkCmdBeginRendering(cmd, &swapRendering);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    if (pipelinesValid_) {
        struct { float exposure; int rawOutput; int frame; float pad; } push{
            settings.exposure, settings.debugView >= 2 ? 1 : 0, int(frameCounter_ & 0x7fffffff), 0.0f};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline_);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    if (settings.showUi) ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
    profiler_.endScope(cmd);

    if (!screenshotPath_.empty()) {
        imageBarrier(cmd, swapImages_[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {full.width, full.height, 1};
        vkCmdCopyImageToBuffer(cmd, swapImages_[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer_, 1, &copy);
        imageBarrier(cmd, swapImages_[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                      VK_ACCESS_2_HOST_READ_BIT);
    } else {
        imageBarrier(cmd, swapImages_[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::saveScreenshot() {
    VK_CHECK(vmaInvalidateAllocation(allocator_, readbackAlloc_, 0, VK_WHOLE_SIZE));
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(allocator_, readbackAlloc_, &info);
    const uint8_t* src = static_cast<const uint8_t*>(info.pMappedData);
    const uint32_t w = readbackExtent_.width, h = readbackExtent_.height;
    const bool bgra = swapFormat_ == VK_FORMAT_B8G8R8A8_SRGB || swapFormat_ == VK_FORMAT_B8G8R8A8_UNORM;

    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        rgb[i * 3 + 0] = src[i * 4 + (bgra ? 2 : 0)];
        rgb[i * 3 + 1] = src[i * 4 + 1];
        rgb[i * 3 + 2] = src[i * 4 + (bgra ? 0 : 2)];
    }
    if (stbi_write_png(screenshotPath_.c_str(), int(w), int(h), 3, rgb.data(), int(w * 3)))
        std::printf("Saved %s (%ux%u)\n", screenshotPath_.c_str(), w, h);
    else
        std::fprintf(stderr, "Failed to write %s\n", screenshotPath_.c_str());
    screenshotPath_.clear();
}
