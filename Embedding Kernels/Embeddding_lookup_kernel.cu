#include <cuda_runtime.h>

__global__ void embedding_lookup_kernel(
    const int* __restrict__ input_ids,        // [B, S]
    const float* __restrict__ embedding_table, // [V, D]
    float* output,                             // [B, S, D]
    int B, int S, int D, int V
) 
{
    int b = blockIdx.x;
    int sd = blockIdx.y * blockDim.x + threadIdx.x;

    if (b < B && sd < S * D) {
        int s = sd / D;
        int d = sd % D;
        int token_id = input_ids[b * S + s];
        if (token_id >= 0 && token_id < V) {
            output[(b * S + s) * D + d] = embedding_table[token_id * D + d];
        } else {
            output[(b * S + s) * D + d] = 0.0f;
        }
    }
}

int total_sd = S * D;
int block_size = 256;
dim3 block_dim(block_size);
dim3 grid_dim(B, (total_sd + block_size - 1) / block_size);

embedding_lookup_kernel<<<grid_dim, block_dim>>>(input_ids, embedding_table, output, B, S, D, V);
