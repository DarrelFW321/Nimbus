cmake_minimum_required(VERSION 3.24)
project(Nimbus LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)
find_program(DXC dxc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(vk_bootstrap
  URL https://github.com/charles-lunarg/vk-bootstrap/archive/refs/tags/v1.3.296.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_Declare(glm
  URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# Header-only / source-only; populated without running their CMake.
FetchContent_Declare(vma
  URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.1.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR _none)
FetchContent_Declare(imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.5.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR _none)
FetchContent_Declare(stb
  URL https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR _none)
FetchContent_Declare(renderdoc_api
  URL https://raw.githubusercontent.com/baldurk/renderdoc/v1.39/renderdoc/api/app/renderdoc_app.h
  DOWNLOAD_NO_EXTRACT TRUE
  SOURCE_SUBDIR _none)

FetchContent_MakeAvailable(glfw vk_bootstrap glm vma imgui stb renderdoc_api)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# ---------------------------------------------------------------------------
# Shaders. GLSL -> SPIR-V via glslc, HLSL -> SPIR-V via DXC.
# The same commands run at runtime when pressing F5 (see shader_compiler.cpp).
# ---------------------------------------------------------------------------
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(GLOB GLSL_SOURCES CONFIGURE_DEPENDS
  ${SHADER_SRC_DIR}/*.comp ${SHADER_SRC_DIR}/*.vert ${SHADER_SRC_DIR}/*.frag)
file(GLOB HLSL_SOURCES CONFIGURE_DEPENDS ${SHADER_SRC_DIR}/*.hlsl)
file(GLOB SHADER_INCLUDES CONFIGURE_DEPENDS ${SHADER_SRC_DIR}/*.glsl ${SHADER_SRC_DIR}/*.hlsli)

set(SPIRV_OUTPUTS)
foreach(src ${GLSL_SOURCES})
  get_filename_component(name ${src} NAME)
  set(out ${SHADER_OUT_DIR}/${name}.spv)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
    COMMAND ${GLSLC} --target-env=vulkan1.3 -O -g -I ${SHADER_SRC_DIR} ${src} -o ${out}
    DEPENDS ${src} ${SHADER_INCLUDES}
    COMMENT "glslc ${name}"
    VERBATIM)
  list(APPEND SPIRV_OUTPUTS ${out})
endforeach()
foreach(src ${HLSL_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  foreach(stage vs ps)
    if(stage STREQUAL vs)
      set(profile vs_6_0)
      set(entry VSMain)
    else()
      set(profile ps_6_0)
      set(entry PSMain)
    endif()
    set(out ${SHADER_OUT_DIR}/${name}.${stage}.spv)
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
      COMMAND ${DXC} -spirv -fspv-target-env=vulkan1.3 -T ${profile} -E ${entry} -Zi -I ${SHADER_SRC_DIR} ${src} -Fo ${out}
      DEPENDS ${src} ${SHADER_INCLUDES}
      COMMENT "dxc ${name}.hlsl (${entry})"
      VERBATIM)
    list(APPEND SPIRV_OUTPUTS ${out})
  endforeach()
endforeach()
add_custom_target(shaders DEPENDS ${SPIRV_OUTPUTS})

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
add_executable(nimbus
  src/main.cpp
  src/renderer.cpp
  src/vk_helpers.cpp
  src/gpu_profiler.cpp
  src/blue_noise.cpp
  src/benchmark.cpp
  src/shader_compiler.cpp
  src/renderdoc_integration.cpp
  src/third_party_impl.cpp)
add_dependencies(nimbus shaders)
target_include_directories(nimbus PRIVATE
  src ${vma_SOURCE_DIR}/include ${stb_SOURCE_DIR} ${renderdoc_api_SOURCE_DIR})
target_link_libraries(nimbus PRIVATE Vulkan::Vulkan glfw vk-bootstrap::vk-bootstrap glm::glm imgui)
target_compile_definitions(nimbus PRIVATE
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLM_FORCE_RADIANS
  SHADER_SRC_DIR="${SHADER_SRC_DIR}"
  SHADER_OUT_DIR="${SHADER_OUT_DIR}"
  GLSLC_PATH="${GLSLC}"
  DXC_PATH="${DXC}"
  $<$<CONFIG:Debug>:ENABLE_VALIDATION=1>)
if(MSVC)
  target_compile_options(nimbus PRIVATE /W4)
else()
  target_compile_options(nimbus PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()
if(MINGW)
  # Static runtime so the exe runs without MinGW DLLs on PATH (e.g. when launched from RenderDoc).
  target_link_options(nimbus PRIVATE -static -static-libgcc -static-libstdc++)
endif()
