#version 460
// Camera-following terrain grid generated from gl_VertexIndex (no vertex buffer).
// The grid is warped (linear + quadratic) so vertices are dense near the camera and sparse towards
// the 80 km horizon, displaced by the tileable heightmap and bent by planet curvature.

#include "common.glsl"

layout(location = 0) out vec3 vWorld;

const ivec2 CORNERS[6] = ivec2[](ivec2(0, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));

void main() {
    int vertsPerSide = int(u.terrain.w);
    int quadsPerSide = vertsPerSide - 1;
    ivec2 v;
    if (hasFlag(FLAG_INDEXED_TERRAIN)) {
        // Indexed draw: one invocation per grid vertex, reused through the post-transform cache.
        v = ivec2(gl_VertexIndex % vertsPerSide, gl_VertexIndex / vertsPerSide);
    } else {
        // Non-indexed: six invocations per quad, every vertex shaded up to six times.
        int quad = gl_VertexIndex / 6;
        v = ivec2(quad % quadsPerSide, quad / quadsPerSide) + CORNERS[gl_VertexIndex % 6];
    }

    vec2 c = vec2(v) / float(quadsPerSide) * 2.0 - 1.0;
    vec2 warped = sign(c) * (0.25 * abs(c) + 0.75 * c * c) * u.terrainOrigin.w;
    vec2 xz = u.terrainOrigin.xy + warped;
    float h = textureLod(heightTex, xz / u.terrain.x, 0.0).r * u.terrain.y;
    vec2 d = xz - u.cameraPos.xz;
    h -= dot(d, d) / (2.0 * EARTH_RADIUS_M);

    vWorld = vec3(xz.x, h, xz.y);
    gl_Position = u.viewProj * vec4(vWorld, 1.0);
}
