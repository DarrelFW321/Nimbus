#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL
// Cloud density model shared by the ray marcher and the cloud shadow map.

#include "common.glsl"

// ---------------------------------------------------------------------------
// Baseline procedural noise (FLAG_NOISE_TEXTURES off): evaluated per sample, deliberately expensive.
// ---------------------------------------------------------------------------
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), f.x),
                   mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
                   mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float valueFbm(vec3 p, int octaves) {
    float sum = 0.0, amp = 0.5, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * valueNoise(p);
        norm += amp;
        p = p * 2.03 + vec3(17.1, 3.7, 9.2);
        amp *= 0.5;
    }
    return sum / norm;
}

// ---------------------------------------------------------------------------
// Density
// ---------------------------------------------------------------------------
const vec3 WIND_DIR = vec3(0.94, 0.0, 0.34);

float heightFraction(vec3 p) {
    return saturate((altitude(p) - u.cloudLayer.x) / (u.cloudLayer.y - u.cloudLayer.x));
}

// Vertical profile by cloud type (0 = stratus, 0.5 = stratocumulus, 1 = cumulus).
float heightGradient(float h, float type) {
    const vec4 stratus       = vec4(0.00, 0.08, 0.12, 0.25);
    const vec4 stratocumulus = vec4(0.00, 0.15, 0.35, 0.60);
    const vec4 cumulus       = vec4(0.00, 0.10, 0.70, 1.00);
    vec4 g = type < 0.5 ? mix(stratus, stratocumulus, type * 2.0) : mix(stratocumulus, cumulus, type * 2.0 - 1.0);
    return smoothstep(g.x, g.y, h) * (1.0 - smoothstep(g.z, g.w, h));
}

// `cheap` skips the detail erosion (used while searching empty space and for distant light samples).
float cloudDensity(vec3 p, float h, bool cheap, float lod) {
    float time = u.cameraPos.w;
    vec3 wp = p + WIND_DIR * (time * u.cloudLight.z + h * 600.0);  // drift + lean tops downwind

    vec2 weather = textureLod(weatherTex, wp.xz * u.cloudNoise.z, 0.0).rg;
    float coverage = saturate(remap(weather.r, 1.0 - u.cloudLayer.z, 1.0, 0.0, 1.0));
    coverage = max(coverage, smoothstep(0.85, 1.0, u.cloudLayer.z) * 0.9);  // overcast at the top end
    if (coverage <= 0.001) return 0.0;

    float base;
    if (hasFlag(FLAG_NOISE_TEXTURES)) {
        vec4 s = textureLod(shapeNoiseTex, wp * u.cloudNoise.x, lod);
        float lowFreq = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
        base = remap(s.r, lowFreq - 1.0, 1.0, 0.0, 1.0);
    } else {
        base = valueFbm(wp * u.cloudNoise.x * 8.0, 5);
        base = saturate(remap(base, 0.3, 0.8, 0.0, 1.0));
    }
    base *= heightGradient(h, weather.g);

    float d = remap(base, 1.0 - coverage, 1.0, 0.0, 1.0) * coverage;
    if (d <= 0.0) return 0.0;

    if (!cheap) {
        vec3 dp = wp * u.cloudNoise.y;
        float detail;
        if (hasFlag(FLAG_NOISE_TEXTURES)) {
            vec3 n = textureLod(detailNoiseTex, dp, lod).rgb;
            detail = n.r * 0.625 + n.g * 0.25 + n.b * 0.125;
        } else {
            detail = valueFbm(dp * 4.0, 3);
        }
        // Wispy at the base, billowy at the top.
        float detailMod = mix(detail, 1.0 - detail, saturate(h * 4.0));
        d = remap(d, detailMod * u.cloudNoise.w, 1.0, 0.0, 1.0);
    }
    return saturate(d) * u.cloudLayer.w;
}

// Segment of the ray inside the cloud shell. Returns (tStart, tEnd); tEnd <= tStart = miss.
vec2 cloudSegment(vec3 ro, vec3 rd, float maxDist) {
    float rBottom = EARTH_RADIUS_M + u.cloudLayer.x;
    float rTop = EARTH_RADIUS_M + u.cloudLayer.y;
    float h = altitude(ro);
    vec2 inner = raySphere(ro, rd, rBottom);
    vec2 outer = raySphere(ro, rd, rTop);
    vec2 ground = raySphere(ro, rd, EARTH_RADIUS_M);
    float tStart, tEnd;

    if (h < u.cloudLayer.x) {           // below the layer
        if (ground.y > ground.x && ground.x > 0.0) return vec2(1.0, 0.0);
        tStart = inner.y;
        tEnd = outer.y;
    } else if (h < u.cloudLayer.y) {    // inside the layer
        tStart = 0.0;
        tEnd = (inner.y > inner.x && inner.x > 0.0) ? inner.x : outer.y;
    } else {                            // above the layer
        if (outer.y <= outer.x || outer.x < 0.0) return vec2(1.0, 0.0);
        tStart = outer.x;
        tEnd = (inner.y > inner.x && inner.x > 0.0) ? inner.x : outer.y;
    }
    return vec2(max(tStart, 0.0), min(tEnd, maxDist));
}

float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// Dual-lobe phase: strong forward scattering (silver lining) plus some back scattering.
float cloudPhase(float cosTheta, float gScale) {
    return mix(henyeyGreenstein(cosTheta, 0.8 * gScale), henyeyGreenstein(cosTheta, -0.3 * gScale), 0.3);
}

#endif
