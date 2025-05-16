#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__device__ __forceinline__ float KL_Loss(float pred, float tar){
    return tar * __logf(tar / pred);

}

__global__ void KL_Loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* pred4 = reinterpret_cast<const float4*>(predictions);
    const float4* tar4  = reinterpret_cast<const float4*>(targets);
    float4* out4        = reinterpret_cast<float4*>(output);
    size_t n4 = n / 4;

    for(size_t i = idx; i < n4; i += stride){
        float4 pred = pred4[i];
        float4 tar = tar4[i];
        float4 result;

        result.x = KL_Loss(pred.x, tar.x);
        result.y = KL_Loss(pred.y, tar.y);
        result.z = KL_Loss(pred.z, tar.z);
        result.w = KL_Loss(pred.w, tar.w);

        out4[i] = result;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) { 
    KL_Loss_kernel<<<((n/4) + BLOCK_DIM - 1)/ BLOCK_DIM, BLOCK_DIM>>>(predictions, targets, output, n); 
}