#include <cuda_runtime.h>

__global__ void leakyRelu_kernel(const float* input, float alpha, float* output, size_t n, size_t m){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n*m) return;
    output[idx] = input[idx] > 0 ? input[idx] : alpha*input[idx];

}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) { 

    int BLOCK_DIM = 1024;
    int GRID_DIM = ((m*n) + BLOCK_DIM - 1) / BLOCK_DIM;

    leakyRelu_kernel<<<GRID_DIM, BLOCK_DIM>>>(input, alpha, output, n, m);

    cudaDeviceSynchronize();
}

//Tensara perf = 210 GFOPS