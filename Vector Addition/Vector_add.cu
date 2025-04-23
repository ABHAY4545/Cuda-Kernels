#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100000
#define CUDA_CHECK(err, msg) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "[%s:%d] CUDA Error: %s\nMessage: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err), (msg)); \
        exit(1); \
    } \
} while (0)


__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

void allocateMemory(float** hostPtr, float** devicePtr, size_t size) {
    *hostPtr = (float*)malloc(size);
    CUDA_CHECK(*hostPtr == NULL ? cudaErrorMemoryAllocation : cudaSuccess, "Host memory allocation failed");

    CUDA_CHECK(cudaMalloc((void**)devicePtr, size), "Device memory allocation failed");
}

void copyToDevice(float* devicePtr, float* hostPtr, size_t size) {
    CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice), "cudaMemcpy Host to Device failed");
}

void copyToHost(float* hostPtr, float* devicePtr, size_t size) {
    CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost), "cudaMemcpy Device to Host failed");
}

int main() {
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    allocateMemory(&h_A, &d_A, size);
    allocateMemory(&h_B, &d_B, size);
    allocateMemory(&h_C, &d_C, size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)i * 2.0f;
    }

    copyToDevice(d_A, h_A, size);
    copyToDevice(d_B, h_B, size); 

    int BLOCK_DIM = 256;
    int GRID_DIM = (N + BLOCK_DIM - 1) / BLOCK_DIM;


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "cudaEventCreate failed");
    CUDA_CHECK(cudaEventCreate(&stop), "cudaEventCreate failed");


    CUDA_CHECK(cudaEventRecord(start), "cudaEventRecord failed");
    vecAdd<<<GRID_DIM, BLOCK_DIM>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop), "cudaEventRecord failed");


    CUDA_CHECK(cudaGetLastError(), "Kernel launch failed");

    copyToHost(h_C, d_C, size);


    CUDA_CHECK(cudaEventSynchronize(stop), "cudaEventSynchronize failed");
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime failed");
    printf("Kernel execution time: %f ms\n", milliseconds);


    CUDA_CHECK(cudaEventDestroy(start), "cudaEventDestroy failed");
    CUDA_CHECK(cudaEventDestroy(stop), "cudaEventDestroy failed");


    CUDA_CHECK(cudaFree(d_A), "cudaFree failed");
    CUDA_CHECK(cudaFree(d_B), "cudaFree failed");
    CUDA_CHECK(cudaFree(d_C), "cudaFree failed");

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition complete. Processed %d elements.\n", N);
    return 0;
}
