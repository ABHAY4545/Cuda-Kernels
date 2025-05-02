#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * m;

    if (idx < total) {
        output[idx] = input[idx] > 0 ? input[idx] : 0.0f;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    relu_kernel<<<blocks, threads_per_block>>>(input, output, n, m);

    cudaDeviceSynchronize();
}   

//Tensara perf = 260 GFOPS