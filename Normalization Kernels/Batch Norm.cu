#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_DIM 512

__global__ void Batch_Norm(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2){
  
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) { 

    dim3 BLOCK(BLOCK_DIM);
    dim3 GRID(B);

    Batch_Norm<<<GRID, BLOCK>>>(X, Y, B, F, D1, D2);

}