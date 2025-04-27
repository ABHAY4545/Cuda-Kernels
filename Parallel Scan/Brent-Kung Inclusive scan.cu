#define BLOCK_DIM 1024

__global__ void inclusive_scan(float* input, float* output, float* partialSums, unsigned int N){

    int segment = blockIdx.x * blockDim.x*2 + threadIdx.x;

    if (segment >= N) return;
    __shared__ float buffer_s[BLOCK_DIM*2];

    buffer_s[threadIdx.x] = (segment < N) ? input[segment] : 0.0f;
    buffer_s[threadIdx.x + BLOCK_DIM] = (segment + BLOCK_DIM < N) ? input[segment + BLOCK_DIM] : 0.0f;
    __syncthreads();

    //Reduction step
    for (int s = 1; s <= BLOCK_DIM; s *= 2){
         int i = (threadIdx.x + 1) * 2 * s -1;

        if ( i < 2 * BLOCK_DIM ){
            buffer_s[i] +=buffer_s[i - s];
        }
        __syncthreads();
    }

    //Post-reduction step 
    for ( int s = BLOCK_DIM/2; s >= 1; s /= 2){
          int i = (threadIdx.x + 1) * 2 * s - 1;

        if( i + s < BLOCK_DIM*2){
         buffer_s[i + s] += buffer_s[i];
        }
        __syncthreads();
    }
   
    if (threadIdx.x == 0){
        partialSums[blockIdx.x] = buffer_s[BLOCK_DIM*2 - 1];
    }
    if (segment < N) output[segment] = buffer_s[threadIdx.x];
    if (segment + BLOCK_DIM < N) output[segment + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];

}