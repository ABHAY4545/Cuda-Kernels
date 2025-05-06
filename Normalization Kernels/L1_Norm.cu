//L1 Norm Vectorized
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void L1_Norm_kernel(const float* __restrict__ X, float* __restrict__ Y, size_t B, size_t D){
    int row = blockIdx.x;
    if (row >= B) return;

    __shared__ float warp_sums[32];
    __shared__ float l1_norm;

    const float* ptr = X + row * D;
    const float4* ptr_4 = reinterpret_cast<const float4*>(ptr);

    float sum = 0.0f;
    int n_float4 = D / 4;

    for(int i = threadIdx.x; i < n_float4; i += BLOCK_DIM){
        float4 v = ptr_4[i];
        sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    for(int offset = 16; offset > 0; offset >>= 1){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((threadIdx.x & 31) == 0)
        warp_sums[threadIdx.x / 32] = sum;

    __syncthreads();

    if (threadIdx.x < 32){
        sum = (threadIdx.x < (BLOCK_DIM / 32)) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0)
            l1_norm = sum;
    }

    __syncthreads();

    float norm = l1_norm;
    for(int d = threadIdx.x; d < D; d += BLOCK_DIM){
        Y[row * D + d] = X[row * D + d] / (norm + 1e-8f);
    }
}


extern "C" void solution(const float* X, float* Y, size_t B, size_t D){
    L1_Norm_kernel<<<B, BLOCK_DIM>>>(X, Y, B, D);
}
