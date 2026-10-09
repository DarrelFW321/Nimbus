#pragma once

#include <array>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include "gpu_profiler.h"
#include "vk_helpers.h"

struct GLFWwindow;

// Feature toggles for A/B profiling. Mirrors the FLAG_* constants in shaders/common.glsl.
enum RenderFlags : int {
    FLAG_NOISE_TEXTURES = 1 << 0,  // 3D noise textures instead of per-sample procedural noise
    FLAG_EARLY_EXIT     = 1 << 1,  // stop marching once transmittance < 1%
    FLAG_ADAPTIVE_STEP  = 1 << 2,  // coarse steps through empty space, fine steps inside clouds
    FLAG_CHEAP_LIGHT    = 1 << 3,  // skip detail noise for the far half of the light march
    FLAG_BLUE_NOISE     = 1 << 4,  // blue-noise ray jitter instead of interleaved gradient noise
    FLAG_INDEXED_TERRAIN = 1 << 5, // indexed grid (vertex reuse) instead of 6 invocations per quad
    FLAG_AERIAL_LUT     = 1 << 6,  // froxel aerial-perspective volume instead of per-pixel integration
    FLAG_ALL            = (1 << 7) - 1,
};

inline constexpr glm::uvec2 kWorkgroupSizes[] = {{8, 8}, {16, 8}, {16, 16}, {8, 4}};
inline constexpr const char* kDebugViewNames[] = {
    "Final", "Clouds only", "Transmittance", "Cloud depth", "Temporal history",
    "Cloud shadow map", "Weather map", "Shape noise slice", "Sky-view LUT"};

struct RenderSettings {
    // Performance
    int   upscaleFactor = 2;  // clouds marched at 1/N x 1/N resolution: 1, 2 or 4
    int   primarySteps  = 96;
    int   lightSteps    = 6;
    int   flags         = FLAG_ALL;
    int   workgroup     = 0;  // index into kWorkgroupSizes
    float temporalBlend = 0.1f;
    bool  vsync         = false;
    // Clouds
    float coverage      = 0.55f;
    float density       = 0.06f;     // extinction per metre at full density
    float layerBottom   = 1500.0f;
    float layerTop      = 4500.0f;
    float shapeTile     = 14000.0f;  // metres covered by one repeat of the shape noise
    float detailTile    = 1000.0f;
    float weatherTile   = 60000.0f;
    float detailErosion = 0.5f; 
    float windSpeed     = 12.0f;
    float ambient       = 0.6f;
    float powder        = 0.5f;
    float fadeDistance  = 35000.0f;
    // Lighting
    float sunAzimuth     = glm::radians(140.0f);
    float sunElevation   = glm::radians(35.0f);
    float sunIlluminance = 12.0f;
    float exposure       = 1.0f;
    // Terrain
    bool  terrain       = true;
    float terrainHeight = 2200.0f;
    // Presentation
    int   debugView = 0;
    bool  showUi    = true;
};

struct CameraState {
    glm::vec3 position{0.0f};
    glm::mat4 viewProj{1.0f};  // reverse-Z projection
};

// std140 mirror of FrameUniforms in shaders/common.glsl.
struct FrameUniforms {
    glm::mat4  viewProj;
    glm::mat4  invViewProj;
    glm::mat4  prevViewProj;
    glm::vec4  cameraPos;
    glm::vec4  sunDir;
    glm::vec4  cloudLayer;
    glm::vec4  cloudNoise;
    glm::vec4  cloudLight;
    glm::vec4  terrain;
    glm::vec4  terrainOrigin;
    glm::vec4  shadowOrigin;
    glm::ivec4 march;
    glm::ivec4 recon;
    glm::ivec4 view;
};

class Renderer {
public:
    void init(GLFWwindow* window, const RenderSettings& settings);
    void shutdown();

    // Records and submits one frame. ImGui::Render() must have been called beforehand.
    void drawFrame(const CameraState& camera, const RenderSettings& settings, float time);

    void requestSwapchainRebuild() { swapchainDirty_ = true; }
    void resetHistory() { historyValid_ = false; }
    bool reloadShaders(std::string& log);
    void requestScreenshot(const std::string& path) { screenshotPath_ = path; }

    const GpuProfiler& profiler() const { return profiler_; }
    VkExtent2D fullExtent() const { return swapchain_.extent; }
    VkExtent2D lowExtent() const { return lowExtent_; }
    const char* deviceName() const { return physicalDevice_.properties.deviceName; }
    uint64_t frameCount() const { return frameCounter_; }

    // CPU copy of the generated heightmap (bilinear, wrapping), excluding planet curvature.
    float terrainHeight(float x, float z, float heightScale) const;

private:
    static constexpr uint32_t kFramesInFlight = 2;

    // Must match the binding numbers in shaders/common.glsl.
    enum Binding : uint32_t {
        B_Uniforms, B_TransmittanceTex, B_TransmittanceImg, B_MultiScatTex, B_MultiScatImg,
        B_SkyViewTex, B_SkyViewImg, B_ShapeTex, B_ShapeImg, B_DetailTex, B_DetailImg,
        B_WeatherTex, B_WeatherImg, B_HeightTex, B_HeightImg, B_ShadowTex, B_ShadowImg,
        B_BlueNoiseTex, B_SceneColorTex, B_SceneDepthTex, B_CloudLowTex, B_CloudLowImg,
        B_CloudDepthTex, B_CloudDepthImg, B_HistoryPrevTex, B_HistoryCurImg, B_HdrTex, B_HdrImg,
        B_AerialTex, B_AerialImg, B_Count
    };

    enum ComputePass : uint32_t {
        CP_Transmittance, CP_MultiScatter, CP_SkyView, CP_NoiseShape, CP_NoiseDetail, CP_Weather,
        CP_Heightmap, CP_CloudShadow, CP_Clouds, CP_Reconstruct, CP_AerialPerspective, CP_Count
    };

    struct Frame {
        VkCommandPool   cmdPool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence         inFlight = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkBuffer        ubo = VK_NULL_HANDLE;
        VmaAllocation   uboAlloc = nullptr;
        void*           uboMapped = nullptr;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    // Setup
    void createDevice(GLFWwindow* window);
    void createSwapchain(bool vsync);
    void destroySwapchain();
    void createSamplers();
    void createDescriptors();
    void createFrames();
    void createStaticResources();
    void destroyStaticResources();
    void createSizedResources(int upscaleFactor);
    void destroySizedResources();
    void writeDescriptors();
    bool createPipelines(std::string& log);
    void createCloudPipeline();
    void destroyPipelines();
    void initImGui();
    void runPrecompute();
    void readBackHeightmap();
    void createTerrainIndexBuffer();

    template <typename F> void submitImmediate(F&& record);
    void handleSettingsChanges(const RenderSettings& settings);
    FrameUniforms buildUniforms(const CameraState& camera, const RenderSettings& settings, float time) const;
    void recordFrame(Frame& frame, uint32_t imageIndex, const RenderSettings& settings);
    void saveScreenshot();

    GLFWwindow* window_ = nullptr;

    vkb::Instance       instance_;
    vkb::PhysicalDevice physicalDevice_;
    vkb::Device         device_;
    VkSurfaceKHR        surface_ = VK_NULL_HANDLE;
    VkQueue             queue_ = VK_NULL_HANDLE;
    uint32_t            queueFamily_ = 0;
    VmaAllocator        allocator_ = nullptr;
    VkCommandPool       immediatePool_ = VK_NULL_HANDLE;

    vkb::Swapchain           swapchain_;
    std::vector<VkImage>     swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkSemaphore> renderFinished_;  // one per swapchain image
    VkFormat                 swapFormat_ = VK_FORMAT_UNDEFINED;
    bool                     swapchainDirty_ = false;
    bool                     currentVsync_ = false;

    VkSampler linearClamp_ = VK_NULL_HANDLE;
    VkSampler linearRepeat_ = VK_NULL_HANDLE;   // tileable noise / weather / height (mipmapped)
    VkSampler skySampler_ = VK_NULL_HANDLE;     // repeat in azimuth, clamp in latitude
    VkSampler nearestClamp_ = VK_NULL_HANDLE;

    // Static resources (generated once at startup / shader reload)
    GpuImage transmittanceLut_, multiScatLut_, skyViewLut_, aerialLut_;
    GpuImage shapeNoise_, detailNoise_, weatherMap_, heightmap_, cloudShadow_, blueNoise_;
    std::vector<uint8_t> blueNoiseData_;
    std::vector<float> heightData_;
    VkBuffer      terrainIndices_ = VK_NULL_HANDLE;
    VmaAllocation terrainIndicesAlloc_ = nullptr;
    uint32_t      terrainIndexCount_ = 0;
    // Resolution-dependent resources
    GpuImage sceneColor_, sceneDepth_, cloudLow_, cloudDepth_, hdr_;
    std::array<GpuImage, 2> history_;
    VkExtent2D lowExtent_{};
    int currentUpscale_ = 0;

    VkDescriptorPool      descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorPool      imguiPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;

    std::array<VkPipeline, CP_Count> computePipelines_{};
    VkPipeline terrainPipeline_ = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline_ = VK_NULL_HANDLE;
    bool       pipelinesValid_ = false;
    int        currentWorkgroup_ = 0;

    std::array<Frame, kFramesInFlight> frames_{};
    uint32_t    frameIndex_ = 0;
    uint64_t    frameCounter_ = 0;
    bool        historyValid_ = false;
    glm::mat4   prevViewProj_{1.0f};
    bool        hasPrevViewProj_ = false;
    GpuProfiler profiler_;

    std::string   screenshotPath_;
    VkBuffer      readbackBuffer_ = VK_NULL_HANDLE;
    VmaAllocation readbackAlloc_ = nullptr;
    VkExtent2D    readbackExtent_{};
};
