#include "shader_compiler.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string quote(const std::string& s) { return "\"" + s + "\""; }

bool run(const std::string& command, const fs::path& errFile, std::ostringstream& log) {
#ifdef _WIN32
    // cmd.exe strips the outermost quotes, so wrap the whole command once more.
    std::string full = "\"" + command + " 2> " + quote(errFile.string()) + "\"";
#else
    std::string full = command + " 2> " + quote(errFile.string());
#endif
    if (std::system(full.c_str()) == 0) return true;
    std::ifstream err(errFile);
    log << err.rdbuf() << "\n";
    return false;
}

}  // namespace

bool compileAllShaders(std::string& log) {
    const fs::path src = SHADER_SRC_DIR, out = SHADER_OUT_DIR;
    const fs::path errFile = out / "compile_errors.txt";
    std::ostringstream messages;
    bool ok = true;

    for (const auto& entry : fs::directory_iterator(src)) {
        const fs::path& file = entry.path();
        std::string ext = file.extension().string();
        std::string name = file.filename().string();

        if (ext == ".comp" || ext == ".vert" || ext == ".frag") {
            std::string cmd = quote(GLSLC_PATH) + " --target-env=vulkan1.3 -O -g -I " + quote(src.string()) + " " +
                              quote(file.string()) + " -o " + quote((out / (name + ".spv")).string());
            ok &= run(cmd, errFile, messages);
        } else if (ext == ".hlsl") {
            std::string stem = file.stem().string();
            struct Stage { const char* suffix; const char* profile; const char* entry; };
            for (Stage s : {Stage{"vs", "vs_6_0", "VSMain"}, Stage{"ps", "ps_6_0", "PSMain"}}) {
                std::string cmd = quote(DXC_PATH) + " -spirv -fspv-target-env=vulkan1.3 -T " + s.profile + " -E " +
                                  s.entry + " -Zi -I " + quote(src.string()) + " " + quote(file.string()) + " -Fo " +
                                  quote((out / (stem + "." + s.suffix + ".spv")).string());
                ok &= run(cmd, errFile, messages);
            }
        }
    }
    log = ok ? "Shaders recompiled." : messages.str();
    return ok;
}
