#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <cuda_runtime.h>

void mergeSequential(float* A, float* B, float*C, int m, int n){

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

    

    return 0;


}
