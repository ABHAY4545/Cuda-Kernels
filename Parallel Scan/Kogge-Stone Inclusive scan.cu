#define BLOCK_DIM 1024

// Base Version
__global__ void inclusive_scan(float* input, float* output, float* partialSums, unsigned int N){

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    output[i] = input[i];
    __syncthreads();

    for (unsigned int s = 1; s <= BLOCK_DIM; s*=2){

        float temp = 0.0f;
        if (threadIdx.x >= s){
            temp = output[i - s];
        }
        __syncthreads();
        if (threadIdx.x >= s){
            output[i] += temp;
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_DIM - 1){
        partialSums[blockIdx.x] = output[i];
    }

}

// Shared Memory

__global__ void inclusive_scan_v2(float* input, float* output, float* partialSums, unsigned int N){
 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if (i >= N) return ;

    __shared__ float buffer_s[BLOCK_DIM];

    buffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for (unsigned int s = 1; s <= BLOCK_DIM; s*=2){
        float temp = 0.0f;
        if(threadIdx.x >= s){
            temp = buffer_s[threadIdx.x - s];
        }
        __syncthreads();
        if(threadIdx.x >= s){
            buffer_s[threadIdx.x] += temp;
        }
    __syncthreads();

    }

    if (threadIdx.x == BLOCK_DIM -1){
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];
    }
    
    output[i] = buffer_s[threadIdx.x];


}


//Double Buffering

__global__ void inclusive_scan_v2(float* input, float* output, float* partialSums, unsigned int N){
 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if (i >= N) return ;

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for(unsigned int s = 1; s <= BLOCK_DIM/2; s*=2){

        if (threadIdx.x >= s){
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - s];
        }
        else{
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;

    }

    if (threadIdx.x == BLOCK_DIM -1 && i < N){
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    output[i] = inBuffer_s[threadIdx.x];

}






__global__ void add_kernel(float* output, float* partialSums, unsigned int N){

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0){
        output[i] += partialSums[blockIdx.x - 1];
    }
}