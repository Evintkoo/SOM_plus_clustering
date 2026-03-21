#include <metal_stdlib>
using namespace metal;

kernel void neighborhood_update(
    device float* neurons         [[buffer(0)]],
    device const float* pt        [[buffer(1)]],
    device const float* influence [[buffer(2)]],
    constant int& mn              [[buffer(3)]],
    constant int& dim             [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if ((int)gid >= mn) return;
    float h = influence[gid];
    for (int d = 0; d < dim; d++) {
        neurons[gid*dim+d] += h * (pt[d] - neurons[gid*dim+d]);
    }
}
