#version 460
// Terrain shading: sun (through atmosphere + cloud shadow map), sky ambient, aerial perspective.

#include "atmosphere.glsl"

layout(location = 0) in vec3 vWorld;
layout(location = 0) out vec4 outColor;

float heightAt(vec2 xz) { return textureLod(heightTex, xz / u.terrain.x, 0.0).r * u.terrain.y; }

void main() {
    vec2 xz = vWorld.xz;
    float e = u.terrain.x / float(textureSize(heightTex, 0).x);
    vec3 n = normalize(vec3(heightAt(xz - vec2(e, 0)) - heightAt(xz + vec2(e, 0)), 2.0 * e,
                            heightAt(xz - vec2(0, e)) - heightAt(xz + vec2(0, e))));
    float height = heightAt(xz);

    // Material by slope and altitude.
    vec3 grass = vec3(0.08, 0.11, 0.04);
    vec3 rock = vec3(0.16, 0.14, 0.12);
    vec3 snow = vec3(0.85, 0.87, 0.90);
    float slope = 1.0 - n.y;
    vec3 albedo = mix(grass, rock, smoothstep(0.15, 0.35, slope));
    float snowLine = 0.72 * u.terrain.y;
    albedo = mix(albedo, snow, smoothstep(snowLine, snowLine + 80.0, height) * smoothstep(0.45, 0.25, slope));

    // Lighting.
    vec3 L = u.sunDir.xyz;
    vec2 shadowUv = (xz - u.shadowOrigin.xy) / u.terrainOrigin.z + 0.5;
    // Fade to unshadowed towards the edge of the shadow map instead of smearing clamped texels.
    vec2 edge = abs(shadowUv - 0.5) * 2.0;
    float shadowFade = 1.0 - smoothstep(0.8, 1.0, max(edge.x, edge.y));
    float cloudShadow = mix(1.0, textureLod(cloudShadowTex, shadowUv, 0.0).r, shadowFade);
    vec3 sun = u.sunDir.w * sunTransmittance(vWorld) * saturate(dot(n, L)) * cloudShadow;
    vec3 ambient = sampleSky(vec3(0.0, 1.0, 0.0)) * (0.5 + 0.5 * n.y) * 2.0;
    vec3 color = albedo / PI * sun + albedo * ambient * 0.8;

    // Aerial perspective between camera and surface.
    vec3 toSurface = vWorld - u.cameraPos.xyz;
    float dist = length(toSurface);
    if (hasFlag(FLAG_AERIAL_LUT)) {
        vec2 screenUv = gl_FragCoord.xy / vec2(u.view.yz);
        vec4 ap = textureLod(aerialTex, vec3(screenUv, aerialKmToSlice(dist * 1e-3)), 0.0);
        color = color * ap.a + ap.rgb;
    } else {
        vec3 T;
        vec3 inscatter = integrateScattering(atmospherePos(u.cameraPos.xyz), toSurface / dist, dist * 1e-3, 8, T);
        color = color * T + inscatter * u.sunDir.w;
    }

    outColor = vec4(color, 1.0);
}
