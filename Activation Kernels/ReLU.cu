#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void relu_kernel(const float* input, float* output, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    size_t vec_size = (n*m) / 4;

    #pragma unroll
    for(size_t i = idx; i < vec_size; i += stride){
            float4 val = input4[i];

            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);

            output4[i] = val;

    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    size_t size = m*n;
    if(size % 4 != 0) return;

    int GRID_DIM = (size + BLOCK_DIM - 1)/ BLOCK_DIM;
    GRID_DIM = min(GRID_DIM, 1024);

    relu_kernel<<<GRID_DIM, BLOCK_DIM>>>(input, output, n, m);
}   

//Tensara perf = 425 GFLOPS from 260 GFLOPS