// Final pass (HLSL, compiled to SPIR-V with DXC): exposure, ACES tonemap, blue-noise dither,
// written to the sRGB swapchain. Binding numbers match src/renderer.h.

[[vk::binding(26, 0)]] [[vk::combinedImageSampler]] Texture2D<float4> hdrTex;
[[vk::binding(26, 0)]] [[vk::combinedImageSampler]] SamplerState hdrSampler;
[[vk::binding(17, 0)]] [[vk::combinedImageSampler]] Texture2D<float> blueNoiseTex;
[[vk::binding(17, 0)]] [[vk::combinedImageSampler]] SamplerState blueNoiseSampler;

struct PushData {
    float exposure;
    int   rawOutput;  // debug views: show values without tonemapping
    int   frame;
    float pad;
};
[[vk::push_constant]] PushData pc;

struct VSOut {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

VSOut VSMain(uint id : SV_VertexID) {
    VSOut o;
    o.uv = float2((id << 1) & 2, id & 2);  // fullscreen triangle
    o.position = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float3 acesFitted(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 linearToSrgb(float3 c) { return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055); }
float3 srgbToLinear(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }

float4 PSMain(VSOut i) : SV_Target {
    float3 hdr = hdrTex.SampleLevel(hdrSampler, i.uv, 0).rgb;
    float3 ldr = pc.rawOutput != 0 ? saturate(hdr) : acesFitted(hdr * pc.exposure);

    // Dither in sRGB space to hide 8-bit banding in the smooth sky gradients.
    uint2 p = uint2(i.position.xy) & 63;
    float noise = frac(blueNoiseTex.Load(int3(p, 0)) + pc.frame * 0.61803398875) - 0.5;
    float3 srgb = linearToSrgb(ldr) + noise / 255.0;
    return float4(srgbToLinear(saturate(srgb)), 1.0);
}
