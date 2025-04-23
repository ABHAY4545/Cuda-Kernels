#define C0 8
#define C1 2

#define BLOCK_DIM 8 // 512 elements 8 power 3
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// Base Stencil Version

__global__ void stencil_kernel_3D(float* in, float* out, size_t N){

    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 ){

                                        //Same Index as output one.
        out[ i * N * N + j * N + k ] =  C0 * in[i * N * N + j * N + k ] +

                                        C1 * (
                                        // Horizontal Elements x-axis
                                               in[i * N * N + j * N + (k - 1)] +
                                               in[i * N * N + j * N + (k + 1)] +
                                        // Vertical Elements in y-axis
                                               in[i * N * N + (j - 1) * N + k ] +
                                               in[i * N * N + (j + 1) * N + k ] +
                                        // Parallel Plane Elements in z-axis
                                               in[(i - 1)* N * N + j * N + k ] +
                                               in[(i + 1)* N * N + j * N + k ] +
                                       );
    }
}

// Tiled Stencil Version

__global__ void tiled_stencil_kernel_3D(float* in, float* out, size_t N){

    size_t i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    size_t j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    size_t k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N){
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 ){
        if(threadIdx.x >=1 && threadIdx.x < blockDim.x && threadIdx.y >=1 &&  threadIdx.y < blockDim.y && threadIdx.z >=1 && threadIdx < blockDim.z){

            out[i * N * N + j * N + k] = C0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                       
                                         C1 * ( 
                                         //Horizontal Elements in x-axis
                                            in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                            in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                        // Vertical Elements in y-axis
                                            in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                            in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                        // Parallel Plane Elements in z-axis
                                            in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                            in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x] 
                                         );
        }    
    }
}
// for Tile = 8 , performance is not worth it.
// Thread Coarsening

// Register Tilling Code