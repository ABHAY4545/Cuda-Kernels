#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void batchNormKernel(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2, float epsilon) {
    __shared__ float shared[2048];
    float* mean_partial = shared;
    float* var_partial = &shared[blockDim.x];

    int tid = threadIdx.x;
    int f = blockIdx.x;
    if (f >= F) return;

    size_t D = D1 * D2;
    size_t BD = B * D;
    float sum = 0.0f;

    for (size_t i = tid; i < BD; i += blockDim.x) {
        size_t b = i / D;
        size_t d = i % D;
        size_t idx = b * F * D + f * D + d;
        sum += X[idx];
    }
    mean_partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            mean_partial[tid] += mean_partial[tid + stride];
        __syncthreads();
    }

    float mean = mean_partial[0] / (float)(BD);

    sum = 0.0f;
    for (size_t i = tid; i < BD; i += blockDim.x) {
        size_t b = i / D;
        size_t d = i % D;
        size_t idx = b * F * D + f * D + d;
        float diff = X[idx] - mean;
        sum += diff * diff;
    }
    var_partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            var_partial[tid] += var_partial[tid + stride];
        __syncthreads();
    }

    float var = var_partial[0] / (float)(BD);
    float inv_std = rsqrtf(var + epsilon);

    for (size_t i = tid; i < BD; i += blockDim.x) {
        size_t b = i / D;
        size_t d = i % D;
        size_t idx = b * F * D + f * D + d;
        Y[idx] = (X[idx] - mean) * inv_std;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    const float epsilon = 1e-5f;
    size_t threads = 1024;
    size_t blocks = F;
    batchNormKernel<<<blocks, threads>>>(X, Y, B, F, D1, D2, epsilon);
}
