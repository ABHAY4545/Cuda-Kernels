#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void fused_relu_dense_bias(
    const float* __restrict__ W,  // [M x K] row-major
    const float* __restrict__ x,  // [K]
    const float* __restrict__ b,  // [M]
    float* __restrict__ y,        // [M]
    int M, int K
){

    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M){
        float sum = 0.0f;
        #pragma unroll
        for(int j = 0; j < K; ++j){
            sum += W[row * K + j] * x[j] ;
        }
        sum += b[row];
        y[row] = fmaxf(0.0f, sum); 
    }     

}
