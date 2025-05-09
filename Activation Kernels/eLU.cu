#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void ELU_Kernel(const float* __restrict__ input, float* __restrict__ output, size_t size, float alpha) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    size_t vec_size = size / 4;

    for (size_t i = thread_id; i < vec_size; i += stride) {
        float4 val = input4[i];

        val.x = val.x > 0.0f ? val.x : alpha * (__expf(val.x) - 1.0f);
        val.y = val.y > 0.0f ? val.y : alpha * (__expf(val.y) - 1.0f);
        val.z = val.z > 0.0f ? val.z : alpha * (__expf(val.z) - 1.0f);
        val.w = val.w > 0.0f ? val.w : alpha * (__expf(val.w) - 1.0f);

        output4[i] = val;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t size = n * m;
    if (size % 4 != 0) return;

    int numBlocks = (size / 4 + BLOCK_DIM - 1) / BLOCK_DIM;
    numBlocks = min(numBlocks, 2048);
    ELU_Kernel<<<numBlocks, BLOCK_DIM>>>(input, output, size, alpha);
}
