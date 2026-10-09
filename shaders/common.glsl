#ifndef COMMON_GLSL
#define COMMON_GLSL
// Shared declarations. FrameUniforms must match src/renderer.h (std140),
// binding numbers must match the Binding enum in src/renderer.h.

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4  viewProj;
    mat4  invViewProj;
    mat4  prevViewProj;
    vec4  cameraPos;      // xyz = world position (m), w = time (s)
    vec4  sunDir;         // xyz = direction towards the sun, w = sun illuminance
    vec4  cloudLayer;     // x = bottom (m), y = top (m), z = coverage, w = density (extinction / m)
    vec4  cloudNoise;     // x = shape freq (1/m), y = detail freq (1/m), z = weather freq (1/m), w = detail erosion
    vec4  cloudLight;     // x = ambient, y = powder, z = wind speed (m/s), w = temporal blend
    vec4  terrain;        // x = tile size (m), y = height scale (m), z = grid spacing (m), w = verts per side
    vec4  terrainOrigin;  // xy = grid centre (world xz), z = cloud shadow map extent (m), w = grid half extent (m)
    vec4  shadowOrigin;   // xy = cloud shadow map centre (world xz), z = cloud fade distance (m), w = unused
    ivec4 march;          // x = primary steps, y = light steps, z = frame index, w = feature flags
    ivec4 recon;          // x = upscale factor, y/z = sub-pixel offset this frame, w = history valid
    ivec4 view;           // x = debug view, y = full width, z = full height, w = terrain enabled
} u;

// Feature flags (march.w). Mirrors RenderFlags in src/renderer.h.
const int FLAG_NOISE_TEXTURES = 1;
const int FLAG_EARLY_EXIT     = 2;
const int FLAG_ADAPTIVE_STEP  = 4;
const int FLAG_CHEAP_LIGHT    = 8;
const int FLAG_BLUE_NOISE     = 16;
const int FLAG_INDEXED_TERRAIN = 32;
const int FLAG_AERIAL_LUT     = 64;
bool hasFlag(int f) { return (u.march.w & f) != 0; }

// Samplers (read) and storage images (write). Only the ones a shader touches end up in its SPIR-V.
layout(set = 0, binding = 1)  uniform sampler2D transmittanceTex;
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D transmittanceImg;
layout(set = 0, binding = 3)  uniform sampler2D multiScatTex;
layout(set = 0, binding = 4, rgba16f) uniform writeonly image2D multiScatImg;
layout(set = 0, binding = 5)  uniform sampler2D skyViewTex;
layout(set = 0, binding = 6, rgba16f) uniform writeonly image2D skyViewImg;
layout(set = 0, binding = 7)  uniform sampler3D shapeNoiseTex;
layout(set = 0, binding = 8, rgba8) uniform writeonly image3D shapeNoiseImg;
layout(set = 0, binding = 9)  uniform sampler3D detailNoiseTex;
layout(set = 0, binding = 10, rgba8) uniform writeonly image3D detailNoiseImg;
layout(set = 0, binding = 11) uniform sampler2D weatherTex;
layout(set = 0, binding = 12, rgba8) uniform writeonly image2D weatherImg;
layout(set = 0, binding = 13) uniform sampler2D heightTex;
layout(set = 0, binding = 14, r32f) uniform writeonly image2D heightImg;
layout(set = 0, binding = 15) uniform sampler2D cloudShadowTex;
layout(set = 0, binding = 16, r32f) uniform writeonly image2D cloudShadowImg;
layout(set = 0, binding = 17) uniform sampler2D blueNoiseTex;
layout(set = 0, binding = 18) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 19) uniform sampler2D sceneDepthTex;
layout(set = 0, binding = 20) uniform sampler2D cloudLowTex;
layout(set = 0, binding = 21, rgba16f) uniform writeonly image2D cloudLowImg;
layout(set = 0, binding = 22) uniform sampler2D cloudDepthTex;
layout(set = 0, binding = 23, r32f) uniform writeonly image2D cloudDepthImg;
layout(set = 0, binding = 24) uniform sampler2D historyPrevTex;
layout(set = 0, binding = 25, rgba16f) uniform writeonly image2D historyCurImg;
layout(set = 0, binding = 26) uniform sampler2D hdrTex;
layout(set = 0, binding = 27, rgba16f) uniform writeonly image2D hdrImg;
layout(set = 0, binding = 28) uniform sampler3D aerialTex;
layout(set = 0, binding = 29, rgba16f) uniform writeonly image3D aerialImg;

const float PI = 3.14159265359;
const float EARTH_RADIUS_M = 6360000.0;
const vec3  EARTH_CENTER_M = vec3(0.0, -EARTH_RADIUS_M, 0.0);

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3 x)  { return clamp(x, 0.0, 1.0); }
vec4  saturate(vec4 x)  { return clamp(x, 0.0, 1.0); }
float remap(float v, float lo, float hi, float newLo, float newHi) {
    return newLo + (v - lo) / (hi - lo) * (newHi - newLo);
}

// World-space view ray through a screen UV / full-resolution pixel centre.
vec3 rayDirectionUv(vec2 uv) {
    vec4 p = u.invViewProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);  // reverse-Z: depth 1 = near plane
    return normalize(p.xyz / p.w - u.cameraPos.xyz);
}
vec3 rayDirection(vec2 pixelCenter) { return rayDirectionUv(pixelCenter / vec2(u.view.yz)); }

// Distance from the camera to the rasterized scene, or a huge value for sky (reverse-Z: depth 0 = far).
float sceneDistance(ivec2 pixel, vec3 rd) {
    float depth = texelFetch(sceneDepthTex, pixel, 0).r;
    if (depth <= 0.0) return 1e9;
    vec2 ndc = (vec2(pixel) + 0.5) / vec2(u.view.yz) * 2.0 - 1.0;
    vec4 p = u.invViewProj * vec4(ndc, depth, 1.0);
    return length(p.xyz / p.w - u.cameraPos.xyz);
}

// Altitude above the spherical ground (m).
float altitude(vec3 p) { return length(p - EARTH_CENTER_M) - EARTH_RADIUS_M; }

// Ray/sphere intersection relative to the planet centre. Returns (near, far); far < near = miss.
// Uses the numerically stable quadratic form; positions are ~6e6 m from the centre.
vec2 raySphere(vec3 ro, vec3 rd, float radius) {
    vec3 oc = ro - EARTH_CENTER_M;
    float b = dot(oc, rd);
    float ocLen = length(oc);
    float c = (ocLen - radius) * (ocLen + radius);
    float disc = b * b - c;
    if (disc < 0.0) return vec2(1.0, -1.0);
    float s = sqrt(disc);
    float q = -b - (b >= 0.0 ? s : -s);  // avoids cancellation
    float t0 = q;
    float t1 = c / q;
    return vec2(min(t0, t1), max(t0, t1));
}

#endif
