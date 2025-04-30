#include <stdio.h>
#define ELMS_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELMS_PER_BLOCK (ELMS_PER_THREAD * THREADS_PER_BLOCK)

__device__ void mergeSequential(float* A, float* B, float*C, int m, int n){

    int i = 0;
    int j = 0;
    int k = 0;

    while ( i < m && j < n){
        if(A[i] < B[j]){
            C[k++] = A[i++];
        }else{
            C[k++] = B[j++];
        }
    }

    while(i < m){
        C[k++] = A[i++];
    }

    while(j < n){
        C[k++] = B[j++];
    }

}


__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k){

    unsigned int iLow = (k > n )?(k - n):0;
    unsigned int iHigh = (m < k)? m:k;
    
    while (true){
        unsigned int i = (iLow + iHigh)/2;
        unsigned int j = k - i;

        if (i > 0 &&  j < n && A[i - 1] > B[j]){
            iHigh = i; 
        } else if( j > 0 && i < m && B[j - 1] > A[i]){
            iLow = i;
        }else{
            return i;
        }
    }
    return iLow;
}


__global__ void merge_krnel(float* A, float* B, float*C, int m, int n){

    unsigned int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELMS_PER_THREAD;

    if ( k < m + n ){
        unsigned int i = coRank(A, B, m, n , k);
        unsigned int j = k - i;
        unsigned int kNext = (k + ELMS_PER_THREAD < m + n) ? (k + ELMS_PER_THREAD): (m + n);
        unsigned int iNext = coRank(A, B, m, n, kNext);
        unsigned int jNext = kNext - iNext;
        mergeSequential(&A[i], &B[j], &C[k], iNext - i, jNext - j);
    }
}
