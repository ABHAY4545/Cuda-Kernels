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


