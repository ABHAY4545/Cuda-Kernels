#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_kernel(const float* input, int dim, float* output, 
                             const size_t* shape, size_t ndim, size_t elements_to_process) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= elements_to_process) return;

    size_t stride = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        if (i >= ndim) return;
        stride *= shape[i];
    }
    
    size_t dim_size = shape[dim];
    if (dim_size == 0) return;

    size_t outer_idx = idx / stride;
    size_t inner_idx = idx % stride;
    size_t start_idx = outer_idx * stride * dim_size + inner_idx;

    float max_val = -FLT_MAX;
    for (size_t i = 0; i < dim_size; i++) {
        size_t in_idx = start_idx + i * stride;
        max_val = fmaxf(max_val, input[in_idx]);
    }

    float exp_sum = 0.0f;
    for (size_t i = 0; i < dim_size; i++) {
        size_t in_idx = start_idx + i * stride;
        exp_sum += expf(input[in_idx] - max_val);
    }

    for (size_t i = 0; i < dim_size; i++) {
        size_t out_idx = start_idx + i * stride;
        output[out_idx] = expf(input[out_idx] - max_val) / exp_sum;
    }
}

extern "C" void solution(const float* input, int dim, float* output, 
                        size_t* shape, size_t ndim) {
    if (!input || !output || !shape || ndim == 0 || dim < 0 || dim >= ndim) {
        return;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }
    if (total_elements == 0) return;

    size_t elements_to_process = total_elements / shape[dim];
    if (shape[dim] == 0) return;

    size_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMemcpy(d_shape, shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    const int threads_per_block = 256;
    const int blocks = (elements_to_process + threads_per_block - 1) / threads_per_block;

    softmax_kernel<<<blocks, threads_per_block>>>(input, dim, output, d_shape, ndim, elements_to_process);
    
    cudaFree(d_shape);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
    }
}