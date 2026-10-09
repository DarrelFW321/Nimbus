#pragma once

// RenderDoc in-application API. Must be initialised before the Vulkan instance is created.
//  * Launched from RenderDoc: the already-injected module is picked up automatically.
//  * `loadLibrary` = true (--renderdoc): loads renderdoc.dll from the default install location.
namespace renderdoc {
bool init(bool loadLibrary);
bool available();
void triggerCapture();        // captures the next frame presented
const char* lastCapturePath();  // empty until a capture has been written
}  // namespace renderdoc
