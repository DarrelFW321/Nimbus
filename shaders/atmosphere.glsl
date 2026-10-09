#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL
// Physically based sky after Hillaire, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique" (EGSR 2020). All distances in this file are kilometres.

#include "common.glsl"

const float ATMO_BOTTOM = 6360.0;
const float ATMO_TOP    = 6460.0;
const vec3  RAYLEIGH_SCATTERING = vec3(5.802, 13.558, 33.1) * 1e-3;
const float RAYLEIGH_SCALE_HEIGHT = 8.0;
const float MIE_SCATTERING = 3.996e-3;
const float MIE_EXTINCTION = 4.440e-3;
const float MIE_SCALE_HEIGHT = 1.2;
const float MIE_G = 0.8;
const vec3  OZONE_ABSORPTION = vec3(0.650, 1.881, 0.085) * 1e-3;
const vec3  GROUND_ALBEDO = vec3(0.3);

const ivec2 TRANSMITTANCE_LUT_SIZE = ivec2(256, 64);
const ivec2 MULTISCAT_LUT_SIZE = ivec2(32, 32);
const ivec2 SKYVIEW_LUT_SIZE = ivec2(192, 108);
const int   AERIAL_LUT_SIZE = 32;
const float AERIAL_MAX_KM = 64.0;  // slices are distributed quadratically up to this distance

// Froxel slice <-> distance. Slice centres are at w = (s + 0.5) / N.
float aerialSliceToKm(float w) { return AERIAL_MAX_KM * w * w; }
float aerialKmToSlice(float km) { return sqrt(saturate(km / AERIAL_MAX_KM)); }

struct Medium {
    vec3 rayleigh;
    float mie;
    vec3 scattering;
    vec3 extinction;
};

Medium sampleMedium(float heightKm) {
    float rayleighDensity = exp(-heightKm / RAYLEIGH_SCALE_HEIGHT);
    float mieDensity = exp(-heightKm / MIE_SCALE_HEIGHT);
    float ozoneDensity = max(0.0, 1.0 - abs(heightKm - 25.0) / 15.0);
    Medium m;
    m.rayleigh = RAYLEIGH_SCATTERING * rayleighDensity;
    m.mie = MIE_SCATTERING * mieDensity;
    m.scattering = m.rayleigh + m.mie;
    m.extinction = m.rayleigh + MIE_EXTINCTION * mieDensity + OZONE_ABSORPTION * ozoneDensity;
    return m;
}

float rayleighPhase(float cosTheta) { return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta); }

float cornetteShanksPhase(float cosTheta, float g) {
    float g2 = g * g;
    float k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
}

// Nearest positive hit of a ray (origin relative to the planet centre) with a sphere, or -1.
float atmoRaySphere(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float s = sqrt(disc);
    float t0 = -b - s, t1 = -b + s;
    if (t0 > 0.0) return t0;
    if (t1 > 0.0) return t1;
    return -1.0;
}

// --- Transmittance LUT parameterisation (Bruneton 2017) ----------------------------------------
void uvToTransmittanceParams(vec2 uv, out float r, out float mu) {
    float H = sqrt(ATMO_TOP * ATMO_TOP - ATMO_BOTTOM * ATMO_BOTTOM);
    float rho = H * uv.y;
    r = sqrt(rho * rho + ATMO_BOTTOM * ATMO_BOTTOM);
    float dMin = ATMO_TOP - r;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec2 transmittanceParamsToUv(float r, float mu) {
    float H = sqrt(max(0.0, ATMO_TOP * ATMO_TOP - ATMO_BOTTOM * ATMO_BOTTOM));
    float rho = sqrt(max(0.0, r * r - ATMO_BOTTOM * ATMO_BOTTOM));
    float disc = r * r * (mu * mu - 1.0) + ATMO_TOP * ATMO_TOP;
    float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    float dMin = ATMO_TOP - r;
    float dMax = rho + H;
    return vec2((d - dMin) / (dMax - dMin), rho / H);
}

vec3 sampleTransmittance(float r, float mu) {
    return textureLod(transmittanceTex, transmittanceParamsToUv(r, mu), 0.0).rgb;
}

vec3 sampleMultiScattering(float r, float mu) {
    vec2 uv = vec2(mu * 0.5 + 0.5, (r - ATMO_BOTTOM) / (ATMO_TOP - ATMO_BOTTOM));
    uv = (uv * vec2(MULTISCAT_LUT_SIZE - 1) + 0.5) / vec2(MULTISCAT_LUT_SIZE);
    return textureLod(multiScatTex, uv, 0.0).rgb;
}

// --- Sky-view LUT parameterisation: world azimuth on U, non-linear latitude on V -----------------
vec2 skyViewUv(vec3 dir) {
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    float v = 0.5 + 0.5 * sign(lat) * sqrt(abs(lat) / (0.5 * PI));
    float az = atan(dir.z, dir.x);
    return vec2(az / (2.0 * PI) + 0.5, v);
}

vec3 skyViewDir(vec2 uv) {
    float vc = uv.y * 2.0 - 1.0;
    float lat = sign(vc) * vc * vc * 0.5 * PI;
    float az = (uv.x - 0.5) * 2.0 * PI;
    return vec3(cos(lat) * cos(az), sin(lat), cos(lat) * sin(az));
}

vec3 sampleSky(vec3 dir) { return textureLod(skyViewTex, skyViewUv(dir), 0.0).rgb; }

// Position in atmosphere space (km, planet centre at origin) for a world position in metres.
vec3 atmospherePos(vec3 worldPos) {
    return vec3(0.0, ATMO_BOTTOM + max(worldPos.y, 0.0) * 1e-3 + 0.002, 0.0);
}

// Transmittance from a world-space point towards the sun (includes planet shadow).
vec3 sunTransmittance(vec3 worldPos) {
    vec3 p = atmospherePos(worldPos);
    float r = length(p);
    return sampleTransmittance(r, dot(u.sunDir.xyz, p / r));
}

// Single scattering + multiple-scattering LUT along a ray, per unit sun illuminance.
vec3 integrateScattering(vec3 ro, vec3 rd, float maxDist, int steps, out vec3 transmittance) {
    float tBottom = atmoRaySphere(ro, rd, ATMO_BOTTOM);
    float tTop = atmoRaySphere(ro, rd, ATMO_TOP);
    float tMax = tBottom > 0.0 ? tBottom : tTop;
    tMax = min(tMax, maxDist);
    transmittance = vec3(1.0);
    if (tMax <= 0.0) return vec3(0.0);

    vec3 sunDir = u.sunDir.xyz;
    float cosTheta = dot(rd, sunDir);
    float phaseR = rayleighPhase(cosTheta);
    float phaseM = cornetteShanksPhase(cosTheta, MIE_G);

    vec3 L = vec3(0.0);
    float dt = tMax / float(steps);
    for (int i = 0; i < steps; ++i) {
        float t = (float(i) + 0.3) * dt;
        vec3 p = ro + rd * t;
        float r = length(p);
        Medium m = sampleMedium(r - ATMO_BOTTOM);
        vec3 sampleT = exp(-m.extinction * dt);

        float sunMu = dot(sunDir, p / r);
        vec3 sunT = sampleTransmittance(r, sunMu);
        float planetShadow = atmoRaySphere(p, sunDir, ATMO_BOTTOM) > 0.0 ? 0.0 : 1.0;
        vec3 S = planetShadow * sunT * (m.rayleigh * phaseR + m.mie * phaseM)
               + sampleMultiScattering(r, sunMu) * m.scattering;

        L += transmittance * (S - S * sampleT) / m.extinction;
        transmittance *= sampleT;
    }
    return L;
}

#endif
