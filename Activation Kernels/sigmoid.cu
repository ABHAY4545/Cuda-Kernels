#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, size_t n, size_t m){

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n && col >= m) return;
    output[row * m + col] = 1 / (1+ ( exp(-input[row* m + col])))
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) { 

    int BLOCK_DIM = 1024;
    int GRID_DIM = (m*n + BLOCK_DIM - 1)   / BLOCK_DIM;
    sigmoid_kernel<<<GRID_DIM, BLOCK_DIM>>>(input, output, n, m);
    cudaDeviceSynchronize();
}