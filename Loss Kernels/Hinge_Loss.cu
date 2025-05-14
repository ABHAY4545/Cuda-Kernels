#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void Hinge_Loss(const float* __restrict__ predictions, const float* __restrict__  targets, float* __restrict__ output, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride  = blockDim.x * gridDim.x;


    for(size_t i = idx; i < n / 4; i += stride){

            const float4 pred4 = reinterpret_cast<const float4*>(predictions)[i];
            const float4 tar4 = reinterpret_cast<const float4*>(targets)[i];
            float4 out4;

            out4.x = fmaxf(0, 1 - pred4.x * tar4.x);
            out4.y = fmaxf(0, 1 - pred4.y * tar4.y);
            out4.z = fmaxf(0, 1 - pred4.z * tar4.z);
            out4.w = fmaxf(0, 1 - pred4.w * tar4.w);

            reinterpret_cast<float4*>(output)[i] = out4;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) { 
    Hinge_Loss<<<((n/4) + BLOCK_DIM - 1)/BLOCK_DIM, BLOCK_DIM>>>(predictions, targets, output, n);   
}