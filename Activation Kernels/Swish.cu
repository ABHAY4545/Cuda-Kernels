#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 1024

__device__ __forceinline__ float Swish(float value){
    return value * (1 / (1 + expf(-value)));
}    

__global__ void Swish_Kernel(const float* __restrict__ input, float* __restrict__ output, size_t vec_size) {  

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    #pragma unroll
    for(size_t i = idx; i < vec_size; i += stride){
        float4 val = input4[i];

        val.x = Swish(val.x);
        val.y = Swish(val.y);
        val.z = Swish(val.z);
        val.w = Swish(val.w);

        output4[i] = val;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {   
    size_t size = n*m;
    if(size % 4 != 0) return;
    size_t vec_size = size / 4;
    int GRID_DIM = (vec_size + BLOCK_DIM - 1)/BLOCK_DIM;
    GRID_DIM = min(GRID_DIM, 1024);
    Swish_Kernel<<<GRID_DIM, BLOCK_DIM>>>(input, output, vec_size);
}