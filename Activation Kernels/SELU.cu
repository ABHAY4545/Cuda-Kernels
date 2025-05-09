#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define alpha  1.67326
#define lambda 1.0507
#define BLOCK_DIM 1024

__global__ void SELU_Kernel(const float* input, float* output, size_t n, size_t m){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    size_t vec_size = (n * m) / 4;

    #pragma unroll
    for(size_t i = idx; i < (n*m)/4; i += stride){
        float4 val = input4[i];

        val.x = lambda * (fmaxf(0, val.x) + fminf(0, alpha*(__expf(val.x) - 1)));
        val.y = lambda * (fmaxf(0, val.y) + fminf(0, alpha*(__expf(val.y) - 1)));
        val.z = lambda * (fmaxf(0, val.z) + fminf(0, alpha*(__expf(val.z) - 1)));
        val.w = lambda * (fmaxf(0, val.w) + fminf(0, alpha*(__expf(val.w) - 1)));

        output4[i] = val;

    }

}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    size_t size = m * n;
    if(size % 4 !=  0) return;

    int GRID_DIM = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    GRID_DIM = min(GRID_DIM, 1024);

    SELU_Kernel<<<GRID_DIM,BLOCK_DIM>>>(input, output, n, m);
}