#include "renderdoc_integration.h"

#include <cstdio>
#include <string>

#include <renderdoc_app.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace renderdoc {
namespace {
RENDERDOC_API_1_6_0* api = nullptr;
std::string capturePath;

pRENDERDOC_GetAPI findGetApi(bool loadLibrary) {
#ifdef _WIN32
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (!mod && loadLibrary) mod = LoadLibraryA("C:\\Program Files\\RenderDoc\\renderdoc.dll");
    if (!mod) return nullptr;
    return reinterpret_cast<pRENDERDOC_GetAPI>(reinterpret_cast<void*>(GetProcAddress(mod, "RENDERDOC_GetAPI")));
#else
    void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (!mod && loadLibrary) mod = dlopen("librenderdoc.so", RTLD_NOW);
    if (!mod) return nullptr;
    return reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
#endif
}
}  // namespace

bool init(bool loadLibrary) {
    pRENDERDOC_GetAPI getApi = findGetApi(loadLibrary);
    if (!getApi || getApi(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&api)) != 1) {
        api = nullptr;
        if (loadLibrary) std::fprintf(stderr, "RenderDoc not found (install it or launch from the RenderDoc UI)\n");
        return false;
    }
    api->SetCaptureFilePathTemplate("captures/nimbus");
    api->MaskOverlayBits(eRENDERDOC_Overlay_None, eRENDERDOC_Overlay_None);  // ImGui shows status instead
    std::printf("RenderDoc API attached\n");
    return true;
}

bool available() { return api != nullptr; }

void triggerCapture() {
    if (api) api->TriggerCapture();
}

const char* lastCapturePath() {
    if (!api) return "";
    uint32_t count = api->GetNumCaptures();
    if (count == 0) return "";
    uint32_t length = 0;
    api->GetCapture(count - 1, nullptr, &length, nullptr);
    capturePath.resize(length);
    api->GetCapture(count - 1, capturePath.data(), &length, nullptr);
    if (!capturePath.empty() && capturePath.back() == '\0') capturePath.pop_back();
    return capturePath.c_str();
}
}  // namespace renderdoc
