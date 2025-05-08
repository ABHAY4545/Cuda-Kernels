#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_DIM 1024


__global__ void ELU_Kernel(const float* input, float* output, size_t n, size_t m, float alpha){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n * m) return;
    output[idx] = input[idx] > 0 ? input[idx] : alpha * (exp(input[idx] ) - 1);

}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {  
    ELU_Kernel<<<(n*m +BLOCK_DIM -1)/BLOCK_DIM, BLOCK_DIM>>>(input, output, n, m, alpha);
}