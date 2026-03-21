extern "C" __global__ void batch_euclidean(
    const float* data,    // [n, dim]
    const float* neurons, // [k, dim]
    float* out,           // [n, k]
    int n, int k, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // sample index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // neuron index
    if (i >= n || j >= k) return;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = data[i*dim+d] - neurons[j*dim+d];
        sum += diff * diff;
    }
    out[i*k+j] = sqrtf(sum);
}
