//L2 Norm Vectorized
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void L2_Norm_kernel(const float* __restrict__ X, float* __restrict__ Y, size_t B, size_t D) {
    int row = blockIdx.x;
    if (row >= B) return;

    __shared__ float warp_sums[32];
    __shared__ float l2_norm;

    int thread_id = threadIdx.x;
    const float* ptr = X + row * D;
    const float4* ptr_4 = reinterpret_cast<const float4*>(ptr);

    float sum = 0.0f;
    int n_float4 = D / 4;
    for (int i = thread_id; i < n_float4; i += BLOCK_DIM) {
        float4 v = ptr_4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if ((thread_id & 31) == 0)
        warp_sums[thread_id / 32] = sum;

    __syncthreads();

    if (thread_id < 32) {
        sum = (thread_id < (BLOCK_DIM / 32)) ? warp_sums[thread_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (thread_id == 0)
            l2_norm = sqrtf(sum + 1e-8f); 
    }

    __syncthreads();

    float norm = l2_norm;
    for (int d = thread_id; d < D; d += BLOCK_DIM)
        Y[row * D + d] = X[row * D + d] / norm;
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    L2_Norm_kernel<<<B, BLOCK_DIM, BLOCK_DIM * sizeof(float)>>>(X, Y, B, D);
}
