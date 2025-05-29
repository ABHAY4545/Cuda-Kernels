#include <cuda_runtime.h>
#define BLOCK_DIM 1024


__global__ void Conv1D_shared(const float* A, const float* B, float* C, size_t N, size_t K) {
    extern __shared__ float shared_B[];

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        shared_B[i] = B[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0f;
    int halfK = K / 2;

    for (int i = 0; i < K; ++i) {
        int input_idx = int(idx) + i - halfK;

        float a_val = 0.0f;
        if (input_idx >= 0 && input_idx < N) {
            a_val = A[input_idx];
        }

        sum += a_val * shared_B[i];
    }

    C[idx] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int num_blocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    size_t shared_mem_bytes = K * sizeof(float);
    Conv1D_shared<<<num_blocks, BLOCK_DIM, shared_mem_bytes>>>(A, B, C, N, K);
}
