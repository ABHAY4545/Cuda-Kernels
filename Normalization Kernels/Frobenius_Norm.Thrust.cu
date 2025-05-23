#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <math.h>

#define BLOCK_DIM 1024

struct square_op {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

__global__ void normalizeKernel(const float* X, float* Y, float norm, size_t size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        Y[i] = X[i] / norm;
    }
}

extern "C" void solution(const float* X, float* Y, size_t size) {
    float h_result;

    thrust::device_ptr<const float> X_ptr(X);
    float sum = thrust::transform_reduce(X_ptr, X_ptr + size, square_op(), 0.0f, thrust::plus<float>());
    h_result = sqrt(sum);

    const int blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    normalizeKernel<<<blocks, BLOCK_DIM>>>(X, Y, h_result, size);
}
