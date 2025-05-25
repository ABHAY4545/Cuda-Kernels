#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void fused_relu_dense_bias_vec4(
    const float* __restrict__ W,  // [M x K] row-major
    const float* __restrict__ x,  // [K]
    const float* __restrict__ b,  // [M]
    float* __restrict__ y,        // [M]
    int M, int K                  // K must be divisible by 4
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M){
        const float4* W_row = reinterpret_cast<const float4*>(&W[row * K]);
        const float4* x_vec = reinterpret_cast<const float4*>(x);

        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < K / 4; ++j) {
            float4 w = W_row[j];
            float4 xv = x_vec[j];
            sum += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
        }

        sum += b[row];
        y[row] = fmaxf(0.0f, sum);
    }
}

int block_size = BLOCK_DIM;
int grid_size = (M + block_size - 1) / block_size;
fused_relu_dense_bias_vec4<<<grid_size, block_size>>>(W, x, b, y, M, K);
