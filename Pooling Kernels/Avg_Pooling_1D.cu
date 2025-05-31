#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_DIM 1024

__global__ void avg_pool_1d(const float* input, int H, int kernel_size, int stride, int padding, float* output, int H_out) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= H_out) return;

    float sum = 0.0f;
    for (int m = 0; m < kernel_size; ++m) {
        int in_idx = out_idx * stride + m - padding;
        if (in_idx >= 0 && in_idx < H) {
            sum += input[in_idx];
        }
    }
    output[out_idx] = sum / kernel_size;
}

extern "C"
void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    avg_pool_1d<<<(H_out + BLOCK_DIM -1)/BLOCK_DIM, BLOCK_DIM>>>(input, H, kernel_size, stride, padding, output, H_out);
}
