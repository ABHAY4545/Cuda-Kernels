#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void L1_Norm_kernel(const float* __restrict__ X, float* __restrict__ Y, size_t B, size_t D) {

    extern __shared__ float partialSum_s[];

    int b = blockIdx.x;      
    int tid = threadIdx.x;
    int offset = b * D;

    float sum = 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        sum += fabsf(X[offset + d]);
    }

    partialSum_s[tid] = sum ;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum_s[tid] += partialSum_s[tid + stride];
        }
        __syncthreads();
    }

    float summ = partialSum_s[0] + 1e-8f;

    for (int d = tid; d < D; d += blockDim.x) {
        Y[offset + d] = X[offset + d] / (summ );
    }
}



extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {

    L1_Norm_kernel<<<B, BLOCK_DIM, BLOCK_DIM * sizeof(float)>>>(X, Y, B, D);
}
