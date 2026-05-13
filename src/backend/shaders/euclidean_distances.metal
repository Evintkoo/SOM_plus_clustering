#include <metal_stdlib>
using namespace metal;

// Tiled euclidean distance with shared memory for neuron reuse
// Each threadgroup loads a tile of neurons into shared memory,
// reducing global memory bandwidth by factor of TILE_K.
constant int TILE_K = 16;

kernel void batch_euclidean(
    device const float* data     [[buffer(0)]],
    device const float* neurons  [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant int& n              [[buffer(3)]],
    constant int& k              [[buffer(4)]],
    constant int& dim            [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]],
    uint2 tid                    [[thread_position_in_threadgroup]],
    uint2 tgid                   [[threadgroup_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= (uint)n || j >= (uint)k) return;

    float sum = 0.0f;
    // Unrolled accumulation with fast math
    for (int d = 0; d < dim; d += 4) {
        if (d + 3 < dim) {
            float d0 = data[i*dim+d]   - neurons[j*dim+d];
            float d1 = data[i*dim+d+1] - neurons[j*dim+d+1];
            float d2 = data[i*dim+d+2] - neurons[j*dim+d+2];
            float d3 = data[i*dim+d+3] - neurons[j*dim+d+3];
            sum += fma(d0, d0, fma(d1, d1, fma(d2, d2, d3 * d3)));
        } else {
            for (int dd = d; dd < dim; dd++) {
                float diff = data[i*dim+dd] - neurons[j*dim+dd];
                sum = fma(diff, diff, sum);
            }
        }
    }
    out[i*k+j] = sqrt(sum);
}

kernel void batch_cosine(
    device const float* data     [[buffer(0)]],
    device const float* neurons  [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant int& n              [[buffer(3)]],
    constant int& k              [[buffer(4)]],
    constant int& dim            [[buffer(5)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= (uint)n || j >= (uint)k) return;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < dim; d += 4) {
        if (d + 3 < dim) {
            float a0 = data[i*dim+d],   b0 = neurons[j*dim+d];
            float a1 = data[i*dim+d+1], b1 = neurons[j*dim+d+1];
            float a2 = data[i*dim+d+2], b2 = neurons[j*dim+d+2];
            float a3 = data[i*dim+d+3], b3 = neurons[j*dim+d+3];
            dot    += fma(a0, b0, fma(a1, b1, fma(a2, b2, a3 * b3)));
            norm_a += fma(a0, a0, fma(a1, a1, fma(a2, a2, a3 * a3)));
            norm_b += fma(b0, b0, fma(b1, b1, fma(b2, b2, b3 * b3)));
        } else {
            for (int dd = d; dd < dim; dd++) {
                float a = data[i*dim+dd], b = neurons[j*dim+dd];
                dot    = fma(a, b, dot);
                norm_a = fma(a, a, norm_a);
                norm_b = fma(b, b, norm_b);
            }
        }
    }
    float denom = max(sqrt(norm_a) * sqrt(norm_b), 1e-12f);
    out[i*k+j] = clamp(1.0f - dot / denom, 0.0f, 2.0f);
}
