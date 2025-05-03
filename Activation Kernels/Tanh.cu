#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, size_t n, size_t m){

    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;
 
    if (row >= n && col >= m) return;

    float val = input[row * m + col];
    float exp_pos = expf(val);
    float exp_neg = expf(-val);

    output[row * m + col] = (exp_pos - exp_neg) / (exp_pos + exp_neg);

}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    

    int BLOCK_DIM = 1024;
    int GRID_DIM = (n*m + BLOCK_DIM - 1)/BLOCK_DIM;

    tanh_kernel<<<GRID_DIM, BLOCK_DIM>>>(input, output, n, m);

    cudaDeviceSynchronize();
}