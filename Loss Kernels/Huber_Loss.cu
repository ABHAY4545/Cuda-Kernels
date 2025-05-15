#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__device__ __forceinline__ float Huber_Loss(float pred, float tar){
    float diff = pred - tar;
    return fabsf(diff) < 1.0f ? 0.5f * diff * diff : fabsf(diff) - 0.5f;
}

__global__ void Huber_Loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = idx; i < n/4; i += stride){
        const float4 pred = reinterpret_cast<const float4*>(predictions)[i];
        const float4 tar = reinterpret_cast<const float4*>(targets)[i];
        float4 result;

        result.x = Huber_Loss(pred.x, tar.x);
        result.y = Huber_Loss(pred.y, tar.y);
        result.z = Huber_Loss(pred.z, tar.z);
        result.w = Huber_Loss(pred.w, tar.w);

        reinterpret_cast<float4*>(output)[i] = result;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) { 
    Huber_Loss_kernel<<<((n/4) + BLOCK_DIM - 1)/ BLOCK_DIM, BLOCK_DIM>>>(predictions, targets, output, n); 
}