#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_DIM 256  

__global__ void RMS_Norm(const float* __restrict__ X, float* __restrict__ Y, size_t B, size_t N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= B) return;

    __shared__ float shared_sum[BLOCK_DIM / 32]; 

    const float* row_ptr = X + row * N;
    float* out_ptr = Y + row * N;

    const float4* input4 = reinterpret_cast<const float4*>(row_ptr);
    size_t n_float4 = N / 4;

    float sum = 0.0f;


    for (size_t i = tid; i < n_float4; i += BLOCK_DIM) {
        float4 val = input4[i];
        sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((tid & 31) == 0) {
        shared_sum[tid >> 5] = sum;
    }

    __syncthreads();

    float total_sum = 0.0f;
    if (tid < (BLOCK_DIM / 32)) {
        total_sum = shared_sum[tid];
    }

    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
        }
    }

    __shared__ float rms;
    if (tid == 0) {
        rms = rsqrtf((total_sum / N) + 1e-8f); 
    }

    __syncthreads();

    float4* output4 = reinterpret_cast<float4*>(out_ptr);
    for (size_t i = tid; i < n_float4; i += BLOCK_DIM) {
        float4 val = input4[i];
        val.x *= rms;
        val.y *= rms;
        val.z *= rms;
        val.w *= rms;
        output4[i] = val;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    RMS_Norm<<<B, BLOCK_DIM>>>(X, Y, B, N);
}
