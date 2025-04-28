#define NUM_BINS 256

//Base Version
__global__ void histogram_kernel(unsigned int* image, unsigned int* bins, unsigned int width, unsigned int height){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    unsigned char b = image[i];
    atomicAdd(&bins[b], 1);
}

// Privatized bins

__global__ void histogram_kernel_v2( unsigned int* image, unsigned int* bins, unsigned int width, unsigned int height){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width*height) return;
    extern __shared__ unsigned int bin_s[NUM_BINS];

    if(threadIdx.x < NUM_BINS)     bin_s[threadIdx.x] = 0;
    __syncthreads();

    unsigned char b = image[i];
    atomicAdd(&bin_s[b], 1);
    __syncthreads();

    if(threadIdx.x < NUM_BINS){
        atomicAdd(&bins[threadIdx.x], bin_s[threadIdx.x]);    
    }
}