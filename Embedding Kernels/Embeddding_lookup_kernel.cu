#include <cuda_runtime.h>

__global__ void embedding_lookup_kernel(
    const int* __restrict__ input_ids,        // [B, S]
    const float* __restrict__ embedding_table, // [V, D]
    float* output,                             // [B, S, D]
    int B, int S, int D, int V
) {
    int b = blockIdx.x;
    int s = threadIdx.y;
    int d = threadIdx.x;

    if (b < B && s < S && d < D) {
        int token_id = input_ids[b * S + s];
        output[(b * S + s) * D + d] = embedding_table[token_id * D + d];
    }
}

dim3 block_dim(D, S);   // threadIdx.x = d, threadIdx.y = s
dim3 grid_dim(B);       // blockIdx.x = b
embedding_lookup_kernel<<<grid_dim, block_dim>>>(input_ids, embedding_table, output, B, S, D, V);
