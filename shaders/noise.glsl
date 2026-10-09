#ifndef NOISE_GLSL
#define NOISE_GLSL
// Tileable gradient (Perlin) and cellular (Worley) noise. Every function takes a `period`
// so the generated textures wrap seamlessly with REPEAT sampling.

// Integer hash (Hoskins), returns [0,1]^3.
vec3 hash33(vec3 p) {
    uvec3 q = uvec3(ivec3(p)) * uvec3(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * uvec3(1597334673u, 3812015801u, 2798796415u);
    return vec3(q) * (1.0 / float(0xffffffffu));
}

// Quintic-interpolated gradient noise with integer period (iq's analytic form). Range ~[-1, 1].
float gradientNoise(vec3 x, float period) {
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    vec3 ga = hash33(mod(p + vec3(0, 0, 0), period)) * 2.0 - 1.0;
    vec3 gb = hash33(mod(p + vec3(1, 0, 0), period)) * 2.0 - 1.0;
    vec3 gc = hash33(mod(p + vec3(0, 1, 0), period)) * 2.0 - 1.0;
    vec3 gd = hash33(mod(p + vec3(1, 1, 0), period)) * 2.0 - 1.0;
    vec3 ge = hash33(mod(p + vec3(0, 0, 1), period)) * 2.0 - 1.0;
    vec3 gf = hash33(mod(p + vec3(1, 0, 1), period)) * 2.0 - 1.0;
    vec3 gg = hash33(mod(p + vec3(0, 1, 1), period)) * 2.0 - 1.0;
    vec3 gh = hash33(mod(p + vec3(1, 1, 1), period)) * 2.0 - 1.0;

    float va = dot(ga, w - vec3(0, 0, 0));
    float vb = dot(gb, w - vec3(1, 0, 0));
    float vc = dot(gc, w - vec3(0, 1, 0));
    float vd = dot(gd, w - vec3(1, 1, 0));
    float ve = dot(ge, w - vec3(0, 0, 1));
    float vf = dot(gf, w - vec3(1, 0, 1));
    float vg = dot(gg, w - vec3(0, 1, 1));
    float vh = dot(gh, w - vec3(1, 1, 1));

    return va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
         + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
         + u.z * u.x * (va - vb - ve + vf) + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Inverted F1 cellular noise: 1 at feature points, falling off between them.
float worleyNoise(vec3 x, float period) {
    vec3 id = floor(x);
    vec3 p = fract(x);
    float minDist = 1e4;
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int xx = -1; xx <= 1; ++xx) {
        vec3 offset = vec3(xx, y, z);
        vec3 feature = hash33(mod(id + offset, period)) + offset;
        vec3 d = p - feature;
        minDist = min(minDist, dot(d, d));
    }
    return 1.0 - minDist;
}

// `uvw` in [0,1)^3, `freq` integer base frequency.
float perlinFbm(vec3 uvw, float freq, int octaves) {
    float gain = exp2(-0.85);
    float amp = 1.0, sum = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * gradientNoise(uvw * freq, freq);
        freq *= 2.0;
        amp *= gain;
    }
    return sum;
}

float worleyFbm(vec3 uvw, float freq) {
    return worleyNoise(uvw * freq, freq) * 0.625
         + worleyNoise(uvw * freq * 2.0, freq * 2.0) * 0.25
         + worleyNoise(uvw * freq * 4.0, freq * 4.0) * 0.125;
}

#endif
