# Benchmark results

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Resolution: 1920x1080
- 300 measured frames per configuration after 90 warm-up frames, identical camera path, GPU timestamp queries

| Configuration | GPU frame (ms) | p95 (ms) | Speed-up | Sky-view LUT | Aerial perspective LUT | Cloud shadow map | Terrain | Clouds (ray march) | Temporal reconstruct | Tonemap + UI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline: procedural noise, full-res march, 128 steps | 40.79 | 43.04 | 1.0x | 0.04 | 0.00 | 1.38 | 0.77 | 38.04 | 0.45 | 0.11 |
| + precomputed 3D noise textures (mipmapped) | 12.19 | 13.05 | 3.3x | 0.05 | 0.00 | 0.27 | 1.17 | 10.18 | 0.41 | 0.11 |
| + early ray termination (T < 1%) | 9.69 | 10.37 | 4.2x | 0.05 | 0.00 | 0.24 | 1.23 | 7.64 | 0.41 | 0.11 |
| + adaptive empty-space stepping | 6.17 | 7.30 | 6.6x | 0.05 | 0.00 | 0.29 | 1.20 | 4.09 | 0.42 | 0.12 |
| + cheap far light-march samples | 6.01 | 6.84 | 6.8x | 0.05 | 0.00 | 0.28 | 1.24 | 3.90 | 0.42 | 0.12 |
| + 96 steps (temporal accumulation hides the noise) | 5.37 | 6.15 | 7.6x | 0.05 | 0.00 | 0.25 | 1.26 | 3.27 | 0.42 | 0.12 |
| + 2x2 temporal reconstruction (1/4 of the rays) | 3.10 | 3.23 | 13.2x | 0.06 | 0.00 | 0.29 | 1.23 | 1.04 | 0.37 | 0.12 |
| + indexed terrain mesh | 3.02 | 3.15 | 13.5x | 0.05 | 0.00 | 0.30 | 1.11 | 1.07 | 0.37 | 0.12 |
| + aerial-perspective froxel LUT (shipping default) | 2.38 | 2.72 | 17.1x | 0.06 | 0.03 | 0.29 | 0.39 | 1.15 | 0.36 | 0.11 |
| Default with 4x4 reconstruction (1/16 of the rays) | 1.73 | 2.00 | 23.6x | 0.08 | 0.02 | 0.23 | 0.40 | 0.54 | 0.35 | 0.11 |
| Default with 16x8 cloud workgroups | 2.56 | 2.73 | 15.9x | 0.10 | 0.03 | 0.27 | 0.45 | 1.24 | 0.36 | 0.11 |
| Default with 16x16 cloud workgroups | 2.58 | 2.75 | 15.8x | 0.09 | 0.02 | 0.27 | 0.46 | 1.26 | 0.36 | 0.11 |
| Default with 8x4 cloud workgroups | 3.05 | 3.39 | 13.4x | 0.06 | 0.02 | 0.26 | 0.44 | 1.80 | 0.35 | 0.11 |
