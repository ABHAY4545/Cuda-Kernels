#include <cuda_runtime.h>

__global__ void squareSumKernel(const float* X, float* partialSums, size_t size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0f;
    
    if (i < size) {
        float x = X[i];
        sdata[tid] = x * x;
    }
    
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

__global__ void normalizeKernel(const float* X, float* Y, float norm, size_t size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        Y[i] = (norm > 0.0f) ? X[i] / norm : X[i];
    }
}

extern "C" void solution(const float* X, float* Y, size_t size) {
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    float* d_partialSums;
    cudaMalloc(&d_partialSums, blocks * sizeof(float));
    
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    
    squareSumKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(X, d_partialSums, size);
    
    float* h_partialSums = new float[blocks];
    cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        sum += h_partialSums[i];
    }
    
    float norm = sqrtf(sum);
    
    delete[] h_partialSums;
    
    normalizeKernel<<<blocks, threadsPerBlock>>>(X, Y, norm, size);
    
    cudaFree(d_partialSums);
}