#pragma once

#include <string>

// Recompiles every shader in SHADER_SRC_DIR into SHADER_OUT_DIR (glslc for GLSL, DXC for HLSL),
// mirroring the CMake rules. Returns false and fills `log` with compiler output on failure.
bool compileAllShaders(std::string& log);
