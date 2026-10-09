#pragma once
// Automated GPU benchmark: replays a fixed camera path once per configuration, each configuration
// adding one optimization on top of the previous one, and writes a Markdown/CSV report.

#include <functional>
#include <string>
#include <vector>

#include "camera.h"
#include "renderer.h"

class Benchmark {
public:
    explicit Benchmark(std::string outputPath);

    // Drives settings/camera/time for the next frame. Returns false once every configuration ran.
    bool update(RenderSettings& settings, FlyCamera& camera, float& time, Renderer& renderer);

    std::string progressText() const;

private:
    struct Config {
        std::string name;
        std::function<void(RenderSettings&)> apply;
    };
    struct Result {
        std::string name;
        std::vector<std::pair<std::string, double>> passMs;  // mean per pass
        double totalMs = 0.0, p95Ms = 0.0;
    };

    void collect(const Renderer& renderer);
    void finishConfig(const Renderer& renderer);
    void writeReport(const Renderer& renderer) const;

    static constexpr int kWarmupFrames = 90;
    static constexpr int kMeasureFrames = 300;

    std::string outputPath_;
    std::vector<Config> configs_;
    std::vector<Result> results_;
    size_t configIndex_ = 0;
    int frame_ = 0;
    std::vector<std::vector<std::pair<std::string, float>>> samples_;  // per measured frame
    std::vector<float> totals_;
};
