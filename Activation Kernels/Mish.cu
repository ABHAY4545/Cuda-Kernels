#include <cuda_runtime.h>
#include <math_constants.h>  

__device__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void mish_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = mish(input[idx]);
    }
}

void mish_forward(const float* d_input, float* d_output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mish_kernel<<<blocks, threads>>>(d_input, d_output, N);
}
