extern "C" __global__ void neighborhood_update(
    float* neurons,           // [m*n, dim]
    const float* data_point,  // [dim]
    const float* influence,   // [m*n]
    int mn, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // neuron index
    if (i >= mn) return;
    float h = influence[i];
    for (int d = 0; d < dim; d++) {
        neurons[i*dim+d] += h * (data_point[d] - neurons[i*dim+d]);
    }
}
