#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void Conv1D(const float* A, const float* B, float* C, size_t N, size_t K){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0f;
    int halfK = K / 2;

    for (size_t i = 0; i < K; ++i) {
        int input_idx = int(idx) + int(i) - halfK;

        float a_val = 0.0f;
        if (input_idx >= 0 && input_idx < N) {
            a_val = A[input_idx];
        }

        sum += a_val * B[i];
    }

    C[idx] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {    
   Conv1D<<<(N + BLOCK_DIM - 1)/BLOCK_DIM, BLOCK_DIM>>>(A, B, C, N, K);
}