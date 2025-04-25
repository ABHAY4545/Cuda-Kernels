#define BLOCK_DIM 1024

// Base version

__global__ void reduction_kernel(float* input, float* partialSums, int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x * 2;

    for (unsigned int s = 0; s <= BLOCK_DIM ; s *=2){

        if(threadIdx.x % stride == 0){
            input[i] += input[i + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        partialSums[blockIdx.x] = input[i]
    }
}

//Coalescing and Minimizing Convergence

 __global__ void reduction_kernel_v2(float* input, float* partialSums, int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x * 2;

    for(unsigned int s = BLOCK_DIM; s > 0; s /= 2){

        if (threadIdx.x < s){
            input[i] += input[i + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        partialSums[blockIdx.x] = input[i];
    }
}



//Shared Memory

__global__ void reduction_kernel_v3(float* input, float* partialSums, int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x * 2;

    __shared__ float float_s[BLOCK_DIM];
    input_s[threadIdx.x] = input[i] + input[BLOCK_DIM];
    __syncthreads();
    for (unsigned int s = BLOCK_DIM/2; s > 0 ; s /=2){

        input_s[threadIdx.x] += input_s[threadIdx.x + s];
    }
    (if threadIdx.x == 0){
        partialSums[blockIdx.x] = input_s[threadIdx.x];
    }


}

// Thread Coarsening

#define COARSE_FACTOR 4

__global__ void reduction_kernel_v4(float* input, float* partialSums, int N){
    unsigned int segment = blockIdx.x * blockDim.x * 2 * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x ;

    __shared__ float input_s[BLOCK_DIM];
    float sum = 0.0f;
    for(unsigned int tile = 0; tile < COARSE_FACTOR ; ++tile){
        unsigned int index = i + tile * BLOCK_DIM;
        if (index < N) {
            sum += input[index];
        }
    }

    input_s[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK_DIM/2; s > 0 ; s /=2){
        if(threadIdx.x < s){

        input_s[threadIdx.x] += input_s[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        partialSums[blockIdx.x] = input_s[threadIdx.x];
    }


}
