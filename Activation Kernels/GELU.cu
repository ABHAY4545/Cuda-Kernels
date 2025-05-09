#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_DIM 1024

__device__ __forceinline__ float GELU(float x) {
    const float sqrt_2_over_pi = 0.7978845608f; 
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

__global__ void GELU_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    size_t vec_size = (n*m)/ 4;

    #pragma unroll
    for(size_t i = idx; i < vec_size; i += stride){
        float4 val = input4[i];

        val.x = GELU(val.x);
        val.y = GELU(val.y);
        val.z = GELU(val.z);
        val.w = GELU(val.w);

        output4[i] = val;
    }

}


extern "C" void solution(const float* input, float* output, size_t n, size_t m) {  
    size_t size = m * n;
    if (size % 4 != 0) return;

    int GRID_DIM = (size + BLOCK_DIM - 1)/ BLOCK_DIM;
    GRID_DIM = min(GRID_DIM, 1024);

    GELU_kernel<<<GRID_DIM,BLOCK_DIM>>>(input, output, n, m);
}