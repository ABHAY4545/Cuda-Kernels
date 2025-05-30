#define MASK_RADIUS 2
#define BLOCK_SIZE 16
#define SHARED_TILE_SIZE (BLOCK_SIZE + MASK_RADIUS * 2)
#define MASK_DIM ((MASK_RADIUS * 2) + 1)


// Base Version
__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void Conv2D_Kernel(const float* input, float* output, int width, int height){

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width){
        float sum =  0.0f;
        for(int mask_Row = 0 ; mask_Row < MASK_DIM ; ++mask_Row){
            for(int mask_Col = 0; mask_Col < MASK_DIM ; ++mask_Col){
                int inRow = outRow - MASK_RADIUS + mask_Row;
                int inCol = outCol - MASK_RADIUS + mask_Col;
            if(inRow < height && inRow >= 0 && inCol < width && inCol >= 0){
                sum += mask_c[mask_Row][mask_Col] * input[inRow * width + inCol];
            }
          }
        }
    output[outRow * width + outCol] = sum;
    }

}


//Tiled Version yet to implement