#define TILE_DIM 32

//Naive GeMM
__global__ void matrix_kerenl(float* A, float* B, float* C, unsigned int N){

    unsigned int row  = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col  = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for(unsigned int i = 0; i < N; ++i){
        sum += A[row * N + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}

//Tiled GeMM

__global__ void tiled_matrix_kerenl(float* A, float* B, float* C, unsigned int N){

    unsigned int row  = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col  = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    
    for(unsigned int tile = 0; tile < N/TILE_DIM; ++tile){

        A_s[threadIdx.y][threadIdx.x] = A[row * N + TILE_DIM * tile + threadIdx.x ];
        B_s[threadIdx.y][threadIdx.x] = B[(TILE_DIM * tile+ threadIdx.y) * N + col];
        __syncthreads();

        for(unsigned int i = 0; i < TILE_DIM; ++i){
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();

        C[row * N + col] = sum;
    }
}

