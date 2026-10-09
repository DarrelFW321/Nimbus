#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "benchmark.h"
#include "camera.h"
#include "renderdoc_integration.h"
#include "renderer.h"

#ifdef _WIN32
// Ask hybrid-GPU laptop drivers for the discrete GPU.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

struct Options {
    int width = 1600, height = 900;
    bool hidden = false;
    bool loadRenderDoc = false;
    std::string benchmarkPath;   // --benchmark [file]
    std::string screenshotPath;  // --screenshot file
    std::string shot = "noon";   // --shot noon|sunset|above
    int screenshotFrames = 240;  // frames to converge temporal accumulation before capture
    int debugView = -1;
    int upscale = -1;
    std::string camera;    // --camera x,y,z,yawDeg,pitchDeg
    std::string sun;       // --sun azimuthDeg,elevationDeg
    float coverage = -1.0f;
    float exposure = -1.0f;
    float flySpeed = 0.0f;  // --fly m/s: move the camera forward in screenshot mode (tests reprojection)
    int exitAfter = -1;     // --exit-after N: quit after N frames (smoke tests)
    std::string sequence;   // --sequence timeofday|flythrough|weather: render a scripted clip
    std::string outDir = "frames";  // --out-dir: where sequence frames are written
    int sequenceFrames = 240;       // --sequence-frames
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--width") o.width = std::atoi(next().c_str());
        else if (a == "--height") o.height = std::atoi(next().c_str());
        else if (a == "--hidden") o.hidden = true;
        else if (a == "--renderdoc") o.loadRenderDoc = true;
        else if (a == "--benchmark") o.benchmarkPath = (i + 1 < argc && argv[i + 1][0] != '-') ? next() : "benchmark_results.md";
        else if (a == "--screenshot") o.screenshotPath = next();
        else if (a == "--shot") o.shot = next();
        else if (a == "--frames") o.screenshotFrames = std::atoi(next().c_str());
        else if (a == "--debug-view") o.debugView = std::atoi(next().c_str());
        else if (a == "--upscale") o.upscale = std::atoi(next().c_str());
        else if (a == "--camera") o.camera = next();
        else if (a == "--sun") o.sun = next();
        else if (a == "--coverage") o.coverage = float(std::atof(next().c_str()));
        else if (a == "--exposure") o.exposure = float(std::atof(next().c_str()));
        else if (a == "--fly") o.flySpeed = float(std::atof(next().c_str()));
        else if (a == "--exit-after") o.exitAfter = std::atoi(next().c_str());
        else if (a == "--sequence") o.sequence = next();
        else if (a == "--out-dir") o.outDir = next();
        else if (a == "--sequence-frames") o.sequenceFrames = std::atoi(next().c_str());
        else std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
    }
    return o;
}

// Camera + lighting presets used for README screenshots. Camera y is metres above the terrain.
void applyShot(const std::string& shot, RenderSettings& s, FlyCamera& cam) {
    if (shot == "sunset") {
        cam.position = {0.0f, 150.0f, 0.0f};
        cam.yaw = glm::radians(-95.0f);
        cam.pitch = glm::radians(9.0f);
        s.sunAzimuth = glm::radians(-90.0f);
        s.sunElevation = glm::radians(3.5f);
        s.coverage = 0.5f;
        s.exposure = 2.2f;
    } else if (shot == "above") {
        cam.position = {0.0f, 6200.0f, 0.0f};
        cam.yaw = glm::radians(30.0f);
        cam.pitch = glm::radians(-12.0f);
        s.sunAzimuth = glm::radians(200.0f);
        s.sunElevation = glm::radians(18.0f);
        s.coverage = 0.6f;
    } else {  // noon: looking across the valley at the main range
        cam.position = {-15000.0f, 120.0f, 9000.0f};
        cam.yaw = glm::radians(150.0f);
        cam.pitch = glm::radians(10.0f);
        s.sunAzimuth = glm::radians(60.0f);
        s.sunElevation = glm::radians(30.0f);
        s.coverage = 0.68f;
    }
}

float smooth01(float t) { return t * t * (3.0f - 2.0f * t); }

// Scripted clips for the README demo videos. `t` runs 0..1 over the clip; returns simulation time
// (time-lapsed, so the wind visibly moves the clouds).
float applySequence(const std::string& name, float t, RenderSettings& s, FlyCamera& cam, const Renderer& r) {
    auto ground = [&](float x, float z) { return r.terrainHeight(x, z, s.terrainHeight); };
    float e = smooth01(t);
    if (name == "timeofday") {
        // Fixed camera across the valley; the sun swings round and sets in front of it.
        cam.position = {-15000.0f, ground(-15000.0f, 9000.0f) + 120.0f, 9000.0f};
        cam.yaw = glm::radians(150.0f);
        cam.pitch = glm::radians(8.0f);
        s.sunAzimuth = glm::radians(glm::mix(70.0f, 152.0f, e));
        s.sunElevation = glm::radians(glm::mix(40.0f, 0.5f, e));
        s.exposure = glm::mix(1.0f, 2.6f, smooth01(std::clamp((t - 0.4f) / 0.6f, 0.0f, 1.0f)));
        s.coverage = 0.68f;
        return 100.0f + t * 900.0f;
    }
    if (name == "weather") {
        // Fixed camera, clear sky building to overcast in a time-lapse.
        cam.position = {-15000.0f, ground(-15000.0f, 9000.0f) + 120.0f, 9000.0f};
        cam.yaw = glm::radians(165.0f);
        cam.pitch = glm::radians(14.0f);
        s.sunAzimuth = glm::radians(70.0f);
        s.sunElevation = glm::radians(28.0f);
        s.coverage = glm::mix(0.42f, 0.84f, e);
        s.windSpeed = 25.0f;
        return 100.0f + t * 1200.0f;
    }
    // flythrough: along the valley, then climb up through the cloud deck and level off above it.
    glm::vec2 start{-15000.0f, 9000.0f};
    glm::vec2 dir{std::sin(glm::radians(150.0f)), -std::cos(glm::radians(150.0f))};
    glm::vec2 xz = start + dir * (16000.0f * t);
    float climb = smooth01(std::clamp((t - 0.3f) / 0.7f, 0.0f, 1.0f));
    float low = ground(xz.x, xz.y) + 180.0f;
    cam.position = {xz.x, std::max(low, glm::mix(ground(start.x, start.y) + 180.0f, 6800.0f, climb)), xz.y};
    cam.yaw = glm::radians(150.0f + 25.0f * e);
    cam.pitch = glm::radians(glm::mix(8.0f, -10.0f, climb));
    s.sunAzimuth = glm::radians(60.0f);
    s.sunElevation = glm::radians(24.0f);
    s.coverage = 0.62f;
    return 100.0f + t * 60.0f;
}

void drawUi(Renderer& renderer, RenderSettings& s, FlyCamera& camera, bool& paused, std::string& status) {
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({400, 0}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Nimbus");

    const GpuProfiler& prof = renderer.profiler();
    VkExtent2D full = renderer.fullExtent(), low = renderer.lowExtent();
    ImGui::Text("%s", renderer.deviceName());
    ImGui::Text("%ux%u  |  clouds marched at %ux%u", full.width, full.height, low.width, low.height);
    ImGui::Text("CPU %.2f ms (%.0f FPS)   GPU %.2f ms", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate, prof.frameSmoothMs());

    if (ImGui::BeginTable("passes", 2, ImGuiTableFlags_SizingStretchProp)) {
        for (const auto& scope : prof.scopes()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(scope.name.c_str());
            ImGui::TableNextColumn();
            char label[32];
            std::snprintf(label, sizeof(label), "%.3f ms", scope.smoothMs);
            float fraction = prof.frameSmoothMs() > 0.0f ? scope.smoothMs / prof.frameSmoothMs() : 0.0f;
            ImGui::ProgressBar(fraction, {-1, 0}, label);
        }
        ImGui::EndTable();
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* upscaleNames[] = {"1x1 (full res)", "2x2 (1/4 rays)", "4x4 (1/16 rays)"};
        int upscaleIndex = s.upscaleFactor == 4 ? 2 : s.upscaleFactor == 2 ? 1 : 0;
        if (ImGui::Combo("Temporal upscale", &upscaleIndex, upscaleNames, 3)) s.upscaleFactor = 1 << upscaleIndex;
        ImGui::SliderInt("Primary steps", &s.primarySteps, 16, 256);
        ImGui::SliderInt("Light steps", &s.lightSteps, 1, 12);
        ImGui::CheckboxFlags("3D noise textures", &s.flags, FLAG_NOISE_TEXTURES);
        ImGui::CheckboxFlags("Early ray termination", &s.flags, FLAG_EARLY_EXIT);
        ImGui::CheckboxFlags("Adaptive stepping", &s.flags, FLAG_ADAPTIVE_STEP);
        ImGui::CheckboxFlags("Cheap far light samples", &s.flags, FLAG_CHEAP_LIGHT);
        ImGui::CheckboxFlags("Blue-noise jitter", &s.flags, FLAG_BLUE_NOISE);
        ImGui::CheckboxFlags("Indexed terrain mesh", &s.flags, FLAG_INDEXED_TERRAIN);
        ImGui::CheckboxFlags("Aerial-perspective LUT", &s.flags, FLAG_AERIAL_LUT);
        static const char* wgNames[] = {"8x8", "16x8", "16x16", "8x4"};
        ImGui::Combo("Cloud workgroup", &s.workgroup, wgNames, IM_ARRAYSIZE(wgNames));
        ImGui::SliderFloat("Temporal blend", &s.temporalBlend, 0.02f, 1.0f);
        ImGui::Checkbox("VSync", &s.vsync);
    }
    if (ImGui::CollapsingHeader("Clouds")) {
        ImGui::SliderFloat("Coverage", &s.coverage, 0.0f, 1.0f);
        ImGui::SliderFloat("Density", &s.density, 0.002f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::DragFloatRange2("Layer (m)", &s.layerBottom, &s.layerTop, 10.0f, 200.0f, 12000.0f);
        ImGui::SliderFloat("Shape tile (m)", &s.shapeTile, 2000.0f, 40000.0f);
        ImGui::SliderFloat("Detail tile (m)", &s.detailTile, 200.0f, 6000.0f);
        ImGui::SliderFloat("Detail erosion", &s.detailErosion, 0.0f, 1.0f);
        ImGui::SliderFloat("Wind (m/s)", &s.windSpeed, 0.0f, 200.0f);
        ImGui::SliderFloat("Ambient", &s.ambient, 0.0f, 3.0f);
        ImGui::SliderFloat("Powder", &s.powder, 0.0f, 1.0f);
        ImGui::Checkbox("Pause time", &paused);
    }
    if (ImGui::CollapsingHeader("Lighting & terrain")) {
        ImGui::SliderAngle("Sun azimuth", &s.sunAzimuth, -180.0f, 180.0f);
        ImGui::SliderAngle("Sun elevation", &s.sunElevation, -10.0f, 90.0f);
        ImGui::SliderFloat("Sun illuminance", &s.sunIlluminance, 0.0f, 40.0f);
        ImGui::SliderFloat("Exposure", &s.exposure, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Terrain", &s.terrain);
        ImGui::SliderFloat("Terrain height (m)", &s.terrainHeight, 0.0f, 4000.0f);
    }
    if (ImGui::CollapsingHeader("Debug & tools", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("View", &s.debugView, kDebugViewNames, IM_ARRAYSIZE(kDebugViewNames));
        if (ImGui::Button("Reload shaders (F5)")) renderer.reloadShaders(status);
        ImGui::SameLine();
        if (ImGui::Button("Reset history")) renderer.resetHistory();
        ImGui::BeginDisabled(!renderdoc::available());
        if (ImGui::Button("RenderDoc capture (F11)")) renderdoc::triggerCapture();
        ImGui::EndDisabled();
        if (!renderdoc::available()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(run with --renderdoc)");
        }
        if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
        ImGui::Text("Camera %.0f, %.0f, %.0f", camera.position.x, camera.position.y, camera.position.z);
        ImGui::TextDisabled("RMB look | WASD/QE move | Shift fast | F1 UI | F2 screenshot");
    }
    ImGui::End();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);
    renderdoc::init(opt.loadRenderDoc);  // before any Vulkan call so RenderDoc can hook the loader

    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (opt.hidden) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);  // exact client size for benchmarks
    }
    GLFWwindow* window = glfwCreateWindow(opt.width, opt.height, "Nimbus | Real-Time Volumetric Renderer", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Window creation failed\n");
        return 1;
    }

    RenderSettings settings;
    FlyCamera camera;
    if (opt.upscale > 0) settings.upscaleFactor = opt.upscale;
    std::unique_ptr<Benchmark> benchmark;
    if (!opt.benchmarkPath.empty()) benchmark = std::make_unique<Benchmark>(opt.benchmarkPath);
    bool screenshotMode = !opt.screenshotPath.empty();
    bool sequenceMode = !opt.sequence.empty();
    if (screenshotMode || benchmark || sequenceMode) settings.showUi = false;
    if (sequenceMode) std::filesystem::create_directories(opt.outDir);
    applyShot(opt.shot, settings, camera);
    if (opt.debugView >= 0) settings.debugView = opt.debugView;
    if (!opt.camera.empty()) {
        float x, y, z, yaw, pitch;
        if (std::sscanf(opt.camera.c_str(), "%f,%f,%f,%f,%f", &x, &y, &z, &yaw, &pitch) == 5) {
            camera.position = {x, y, z};
            camera.yaw = glm::radians(yaw);
            camera.pitch = glm::radians(pitch);
        }
    }
    if (!opt.sun.empty()) {
        float az, el;
        if (std::sscanf(opt.sun.c_str(), "%f,%f", &az, &el) == 2) {
            settings.sunAzimuth = glm::radians(az);
            settings.sunElevation = glm::radians(el);
        }
    }
    if (opt.coverage >= 0.0f) settings.coverage = opt.coverage;
    if (opt.exposure > 0.0f) settings.exposure = opt.exposure;

    Renderer renderer;
    int exitCode = 0;
    try {
        renderer.init(window, settings);
        if (opt.camera.empty() && settings.terrain)
            camera.position.y += renderer.terrainHeight(camera.position.x, camera.position.z, settings.terrainHeight);
        glfwSetWindowUserPointer(window, &renderer);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
            static_cast<Renderer*>(glfwGetWindowUserPointer(w))->requestSwapchainRebuild();
        });

        std::string status;
        bool paused = false;
        float time = 0.0f;
        int frame = 0;
        int screenshotIndex = 0;
        double last = glfwGetTime();

        while (!glfwWindowShouldClose(window) && frame != opt.exitAfter) {
            glfwPollEvents();
            double now = glfwGetTime();
            float dt = float(now - last);
            last = now;

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (benchmark) {
                if (!benchmark->update(settings, camera, time, renderer)) break;
            } else if (sequenceMode) {
                // Warm-up frames at t = 0 let the temporal history converge before recording.
                constexpr int kWarmup = 60;
                int index = frame - kWarmup;
                if (index >= opt.sequenceFrames) break;
                float t = index <= 0 ? 0.0f : float(index) / float(opt.sequenceFrames - 1);
                time = applySequence(opt.sequence, t, settings, camera, renderer);
                if (index >= 0) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/frame_%04d.png", opt.outDir.c_str(), index);
                    renderer.requestScreenshot(path);
                }
            } else if (screenshotMode) {
                time = 60.0f + float(frame) / 60.0f;  // deterministic
                camera.position += camera.forward() * (opt.flySpeed / 60.0f);
                if (frame == opt.screenshotFrames) renderer.requestScreenshot(opt.screenshotPath);
                if (frame > opt.screenshotFrames) break;
            } else {
                if (!paused) time += dt;
                ImGuiIO& io = ImGui::GetIO();
                camera.update(window, dt, io.WantCaptureMouse, io.WantCaptureKeyboard);
                if (settings.terrain) {  // keep the camera above the ground
                    float ground = renderer.terrainHeight(camera.position.x, camera.position.z, settings.terrainHeight);
                    camera.position.y = std::max(camera.position.y, ground + 20.0f);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) settings.showUi = !settings.showUi;
                if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) renderer.reloadShaders(status);
                if (ImGui::IsKeyPressed(ImGuiKey_F11, false)) renderdoc::triggerCapture();
                if (ImGui::IsKeyPressed(ImGuiKey_F2, false)) {
                    std::filesystem::create_directories("screenshots");
                    std::string path = "screenshots/shot_" + std::to_string(screenshotIndex++) + ".png";
                    renderer.requestScreenshot(path);
                    status = "Saved " + path;
                }
                if (renderdoc::available() && *renderdoc::lastCapturePath())
                    status = std::string("RenderDoc capture: ") + renderdoc::lastCapturePath();
                if (settings.showUi) drawUi(renderer, settings, camera, paused, status);
            }
            ImGui::Render();

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            float aspect = h > 0 ? float(w) / float(h) : 1.0f;
            renderer.drawFrame({camera.position, camera.viewProj(aspect)}, settings, time);
            ++frame;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        exitCode = 1;
    }

    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}
