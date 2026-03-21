#include <metal_stdlib>
using namespace metal;

kernel void batch_euclidean(
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
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = data[i*dim+d] - neurons[j*dim+d];
        sum += diff * diff;
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
    for (int d = 0; d < dim; d++) {
        float a = data[i*dim+d], b = neurons[j*dim+d];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    float denom = max(sqrt(norm_a) * sqrt(norm_b), 1e-12f);
    float cosine_sim = dot / denom;
    out[i*k+j] = clamp(1.0f - cosine_sim, 0.0f, 2.0f);
}
