#pragma once

#include <cstdint>
#include <vector>

// Tileable blue-noise threshold map generated with Ulichney's void-and-cluster algorithm (1993).
// Returns size*size values in [0, 255]; each value appears equally often.
std::vector<uint8_t> generateBlueNoise(int size, uint32_t seed = 1);
