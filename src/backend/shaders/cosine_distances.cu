extern "C" __global__ void batch_cosine(
    const float* data,    // [n, dim]
    const float* neurons, // [k, dim]
    float* out,           // [n, k]
    int n, int k, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= k) return;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < dim; d++) {
        float a = data[i*dim+d], b = neurons[j*dim+d];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    float denom = fmaxf(sqrtf(norm_a) * sqrtf(norm_b), 1e-12f);
    float cosine_sim = dot / denom;
    out[i*k+j] = fmaxf(0.0f, fminf(2.0f, 1.0f - cosine_sim));
}
