#define BLOCK_DIM 1024


//Double Buffering

__global__ void exclusive_scan(float* input, float* output, float* partialSums, unsigned int N){
 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ;
    if (i >= N) return ;

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;

    if (threadIdx.x == 0 ){
        inBuffer_s[threadIdx] = 0.0f;
    } else{
        inBuffer_s[threadIdx.x] = input[i - 1];
    }


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
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x] + input[i];
    }
    output[i] = inBuffer_s[threadIdx.x] + input[i];

}





__global__ void add_kernel(float* output, float* partialSums, unsigned int N){

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0){
        output[i] += partialSums[blockIdx.x ];
    }
}