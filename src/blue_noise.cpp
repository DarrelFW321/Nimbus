#include "blue_noise.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

// Energy field: sum of toroidal Gaussians centred on every set pixel.
class EnergyField {
public:
    EnergyField(int size, float sigma) : size_(size), kernel_(size * size), energy_(size * size, 0.0f) {
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                int dx = std::min(x, size - x), dy = std::min(y, size - y);
                kernel_[y * size + x] = std::exp(-float(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            }
    }

    void splat(int index, float sign) {
        int px = index % size_, py = index / size_;
        for (int y = 0; y < size_; ++y) {
            int ky = (y - py + size_) % size_;
            for (int x = 0; x < size_; ++x) {
                int kx = (x - px + size_) % size_;
                energy_[y * size_ + x] += sign * kernel_[ky * size_ + kx];
            }
        }
    }

    // Tightest cluster = highest energy among set pixels; largest void = lowest among unset.
    int find(const std::vector<uint8_t>& pattern, bool wantSet, bool highest) const {
        int best = -1;
        float bestValue = highest ? -1e30f : 1e30f;
        for (int i = 0; i < size_ * size_; ++i) {
            if (bool(pattern[i]) != wantSet) continue;
            if (highest ? energy_[i] > bestValue : energy_[i] < bestValue) {
                bestValue = energy_[i];
                best = i;
            }
        }
        return best;
    }

private:
    int size_;
    std::vector<float> kernel_;
    std::vector<float> energy_;
};

}  // namespace

std::vector<uint8_t> generateBlueNoise(int size, uint32_t seed) {
    const int n = size * size;
    const float sigma = 1.9f;
    std::mt19937 rng(seed);

    // 1. Random initial binary pattern (~10% ones), relaxed until stable.
    std::vector<uint8_t> initial(n, 0);
    int ones = n / 10;
    for (int placed = 0; placed < ones;) {
        int i = int(rng() % uint32_t(n));
        if (!initial[i]) { initial[i] = 1; ++placed; }
    }
    EnergyField field(size, sigma);
    for (int i = 0; i < n; ++i) if (initial[i]) field.splat(i, 1.0f);
    for (int iter = 0; iter < n; ++iter) {
        int cluster = field.find(initial, true, true);
        initial[cluster] = 0;
        field.splat(cluster, -1.0f);
        int voidIdx = field.find(initial, false, false);
        initial[voidIdx] = 1;
        field.splat(voidIdx, 1.0f);
        if (voidIdx == cluster) break;
    }

    std::vector<int> rank(n, 0);

    // 2. Phase 1: remove tightest clusters from the initial pattern, ranking downwards.
    {
        std::vector<uint8_t> pattern = initial;
        EnergyField f(size, sigma);
        for (int i = 0; i < n; ++i) if (pattern[i]) f.splat(i, 1.0f);
        for (int r = ones - 1; r >= 0; --r) {
            int cluster = f.find(pattern, true, true);
            pattern[cluster] = 0;
            f.splat(cluster, -1.0f);
            rank[cluster] = r;
        }
    }

    // 3. Phases 2+3: fill the largest voids, ranking upwards.
    {
        std::vector<uint8_t> pattern = initial;
        EnergyField f(size, sigma);
        for (int i = 0; i < n; ++i) if (pattern[i]) f.splat(i, 1.0f);
        for (int r = ones; r < n; ++r) {
            int voidIdx = f.find(pattern, false, false);
            pattern[voidIdx] = 1;
            f.splat(voidIdx, 1.0f);
            rank[voidIdx] = r;
        }
    }

    std::vector<uint8_t> out(n);
    for (int i = 0; i < n; ++i) out[i] = uint8_t((rank[i] * 256) / n);
    return out;
}
