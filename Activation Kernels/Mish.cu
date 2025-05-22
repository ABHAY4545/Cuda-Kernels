#include <cuda_runtime.h>
#include <math_constants.h>

__device__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void mish_kernel_float4(const float4* input, float4* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int float4_idx = idx;

    if (float4_idx * 4 < N) {
        float4 in = input[float4_idx];
        float4 out;
        out.x = mish(in.x);
        out.y = mish(in.y);
        out.z = mish(in.z);
        out.w = mish(in.w);
        output[float4_idx] = out;
    }
}

void mish_forward_float4(const float* d_input, float* d_output, int N) {
    int threads = 256;
    int float4_N = (N + 3) / 4;
    int blocks = (float4_N + threads - 1) / threads;

    mish_kernel_float4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(d_input),
        reinterpret_cast<float4*>(d_output),
        N
    );
}
