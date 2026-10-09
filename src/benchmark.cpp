#include "benchmark.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

Benchmark::Benchmark(std::string outputPath) : outputPath_(std::move(outputPath)) {
    auto make = [](int flags, int steps, int upscale, int workgroup) {
        return [=](RenderSettings& s) {
            s = RenderSettings{};
            s.vsync = false;
            s.showUi = false;
            s.flags = flags;
            s.primarySteps = steps;
            s.upscaleFactor = upscale;
            s.workgroup = workgroup;
        };
    };
    // Cumulative: each row keeps every optimization above it.
    int f = 0;
    configs_.push_back({"Baseline: procedural noise, full-res march, 128 steps", make(f, 128, 1, 0)});
    f |= FLAG_NOISE_TEXTURES | FLAG_BLUE_NOISE;
    configs_.push_back({"+ precomputed 3D noise textures (mipmapped)", make(f, 128, 1, 0)});
    f |= FLAG_EARLY_EXIT;
    configs_.push_back({"+ early ray termination (T < 1%)", make(f, 128, 1, 0)});
    f |= FLAG_ADAPTIVE_STEP;
    configs_.push_back({"+ adaptive empty-space stepping", make(f, 128, 1, 0)});
    f |= FLAG_CHEAP_LIGHT;
    configs_.push_back({"+ cheap far light-march samples", make(f, 128, 1, 0)});
    configs_.push_back({"+ 96 steps (temporal accumulation hides the noise)", make(f, 96, 1, 0)});
    configs_.push_back({"+ 2x2 temporal reconstruction (1/4 of the rays)", make(f, 96, 2, 0)});
    f |= FLAG_INDEXED_TERRAIN;
    configs_.push_back({"+ indexed terrain mesh", make(f, 96, 2, 0)});
    f |= FLAG_AERIAL_LUT;
    configs_.push_back({"+ aerial-perspective froxel LUT (shipping default)", make(f, 96, 2, 0)});
    configs_.push_back({"Default with 4x4 reconstruction (1/16 of the rays)", make(f, 96, 4, 0)});
    for (int wg = 1; wg < int(std::size(kWorkgroupSizes)); ++wg) {
        char name[64];
        std::snprintf(name, sizeof(name), "Default with %ux%u cloud workgroups", kWorkgroupSizes[wg].x,
                      kWorkgroupSizes[wg].y);
        configs_.push_back({name, make(f, 96, 2, wg)});
    }
}

std::string Benchmark::progressText() const {
    if (configIndex_ >= configs_.size()) return "Benchmark complete";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "Benchmark %zu/%zu: %s", configIndex_ + 1, configs_.size(),
                  configs_[configIndex_].name.c_str());
    return buf;
}

bool Benchmark::update(RenderSettings& settings, FlyCamera& camera, float& time, Renderer& renderer) {
    if (configIndex_ >= configs_.size()) return false;

    if (frame_ == 0) {
        configs_[configIndex_].apply(settings);
        renderer.resetHistory();
        samples_.clear();
        totals_.clear();
        std::printf("[%zu/%zu] %s\n", configIndex_ + 1, configs_.size(), configs_[configIndex_].name.c_str());
    } else if (frame_ > kWarmupFrames) {
        collect(renderer);
    }

    // Deterministic slow fly-through of the valley: broken cumulus above, mountains ahead,
    // 350 m above the ground.
    float t = float(frame_) / float(kWarmupFrames + kMeasureFrames);
    camera.yaw = glm::radians(150.0f - 30.0f * t);
    camera.pitch = glm::radians(8.0f);
    float x = -15000.0f + 2500.0f * t, z = 9000.0f + 3000.0f * t;
    camera.position = {x, renderer.terrainHeight(x, z, settings.terrainHeight) + 350.0f, z};
    time = 120.0f + float(frame_) / 60.0f;

    if (++frame_ > kWarmupFrames + kMeasureFrames) {
        finishConfig(renderer);
        frame_ = 0;
        if (++configIndex_ >= configs_.size()) {
            writeReport(renderer);
            return false;
        }
    }
    return true;
}

void Benchmark::collect(const Renderer& renderer) {
    std::vector<std::pair<std::string, float>> frame;
    const auto& scopes = renderer.profiler().scopes();
    for (const auto& name : renderer.profiler().lastFrameOrder()) {
        auto it = std::find_if(scopes.begin(), scopes.end(), [&](const auto& s) { return s.name == name; });
        if (it != scopes.end()) frame.emplace_back(name, it->lastMs);
    }
    samples_.push_back(std::move(frame));
    totals_.push_back(renderer.profiler().frameMs());
}

void Benchmark::finishConfig(const Renderer& renderer) {
    Result r;
    r.name = configs_[configIndex_].name;
    std::map<std::string, double> sums;
    for (const auto& frame : samples_)
        for (const auto& [name, ms] : frame) sums[name] += ms;
    for (const auto& scope : renderer.profiler().scopes())
        r.passMs.emplace_back(scope.name, sums[scope.name] / double(std::max<size_t>(1, samples_.size())));

    std::vector<float> sorted = totals_;
    std::sort(sorted.begin(), sorted.end());
    for (float v : sorted) r.totalMs += v;
    r.totalMs /= double(std::max<size_t>(1, sorted.size()));
    r.p95Ms = sorted.empty() ? 0.0 : sorted[size_t(double(sorted.size() - 1) * 0.95)];
    std::printf("    GPU frame %.3f ms (p95 %.3f ms)\n", r.totalMs, r.p95Ms);
    results_.push_back(std::move(r));
}

void Benchmark::writeReport(const Renderer& renderer) const {
    // Columns in pipeline order, taken from the last configuration (which runs every pass).
    std::vector<std::string> passes;
    if (!samples_.empty())
        for (const auto& [name, ms] : samples_.back()) passes.push_back(name);
    for (const auto& scope : renderer.profiler().scopes())
        if (std::find(passes.begin(), passes.end(), scope.name) == passes.end()) passes.push_back(scope.name);
    auto passValue = [](const Result& r, const std::string& name) {
        for (const auto& [n, ms] : r.passMs)
            if (n == name) return ms;
        return 0.0;
    };

    std::ostringstream md;
    md << "# Benchmark results\n\n";
    md << "- GPU: " << renderer.deviceName() << "\n";
    md << "- Resolution: " << renderer.fullExtent().width << "x" << renderer.fullExtent().height << "\n";
    md << "- " << kMeasureFrames << " measured frames per configuration after " << kWarmupFrames
       << " warm-up frames, identical camera path, GPU timestamp queries\n\n";
    md << "| Configuration | GPU frame (ms) | p95 (ms) | Speed-up |";
    for (const auto& p : passes) md << " " << p << " |";
    md << "\n|---|---:|---:|---:|";
    for (size_t i = 0; i < passes.size(); ++i) md << "---:|";
    md << "\n";
    char buf[64];
    for (const Result& r : results_) {
        md << "| " << r.name;
        std::snprintf(buf, sizeof(buf), " | %.2f | %.2f | %.1fx |", r.totalMs, r.p95Ms, results_[0].totalMs / r.totalMs);
        md << buf;
        for (const auto& p : passes) {
            std::snprintf(buf, sizeof(buf), " %.2f |", passValue(r, p));
            md << buf;
        }
        md << "\n";
    }

    std::ofstream(outputPath_) << md.str();
    std::ofstream csv(outputPath_ + ".csv");
    csv << "configuration,gpu_frame_ms,p95_ms";
    for (const auto& p : passes) csv << "," << p;
    csv << "\n";
    for (const Result& r : results_) {
        csv << "\"" << r.name << "\"," << r.totalMs << "," << r.p95Ms;
        for (const auto& p : passes) csv << "," << passValue(r, p);
        csv << "\n";
    }
    std::printf("\n%s\nWrote %s and %s.csv\n", md.str().c_str(), outputPath_.c_str(), outputPath_.c_str());
}
