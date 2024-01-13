#include <stdio.h>

/*
Approach:

Each thread computes a single dot product
A block is defined by a square area in the output

A: i x j
B: j x k

Approach:

1. populate shared memory for the block 

Problem:  we have block_side**2 threads, but block_side * J elements to copy into
shared memory.  

the (threadIdx.y, threadIdx.x)'th thread will copy subchunk x from A[y][d]

but, we still would like each thread to carry out a full dot product?


   */
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))


__global__ void matmul(const int *A, const int *B, int *C, const int block_side, 
    const int I, const int J, const int K){
  extern __shared__ int buf[];
  // extern __shared__ int Arows[];
  // extern __shared__ int Bcols[]; 

  int *Arows = buf; // [0, block_side * J)
  int *Bcols = buf + block_side * J; // [block_side * J, 2 * block_side * J)

  // column and row of C (also, row of A, column of B)
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= I || x >= K)
    return;

  // printf("(x, y) = %d, %d\n", x, y);

  int chunk_size;
  if (J % block_side == 0)
    chunk_size = J / block_side;
  else
    chunk_size = J / block_side + 1;

  int j_beg = MIN(chunk_size * threadIdx.x, J);
  int j_end = MIN(j_beg + chunk_size, J);

  for (int j = j_beg; j != j_end; j++)
    Arows[threadIdx.y * J + j] = A[y * J + j];
  
  j_beg = MIN(chunk_size * threadIdx.y, J);
  j_end = MIN(j_beg + chunk_size, J);

  for (int j = j_beg; j != j_end; j++)  
    Bcols[threadIdx.x * J + j] = B[j * K + x];
  __syncthreads();

  if (y == 0) 
    for (int j = 0; j != J; j++)
      if (Bcols[threadIdx.x * J + j] != B[j * K + x])
        printf("Bcols != B at [%d][%d]: %d != %d\n",
            j, x, Bcols[threadIdx.x * J + j], B[j * K + x]);

  if (x == 0)
    for (int j = 0; j != J; j++)
        if (Arows[threadIdx.y * J + j] != A[y * J + j])
          printf("Arows != A at [%d][%d]: %d != %d\n", 
              y, j, Arows[threadIdx.y * J + j], A[y * J + j]);

  int dp = 0;
  for (int j = 0; j != J; j++){
    dp += Arows[threadIdx.y * J + j] * Bcols[threadIdx.x * J + j];
    // dp += Arows[threadIdx.y * J + j];
    // dp += Bcols[threadIdx.x * J + j];
  }

  /*
  if (dp == 0) {
    printf("dp = 0: (x, y) = %d, %d\n", x, y);
    for (int d = 0; d != J; d++)
      printf("%d, ", Arows[threadIdx.y * J + d]);
    printf("\n");
    for (int d = 0; d != J; d++)
      printf("%d, ", Bcols[threadIdx.x * J + d]);
    printf("\n");
  }
  */
  // printf("%d\n", dp);
  C[y * K + x] = dp;
  // C[y * K + x] = 1;
  // C[0] = 1;
}



int main(){
  const int I = 50;
  const int J = 50;
  const int K = 50;
  const int block_side = 32;

  int *A, *B, *C, *dA, *dB, *dC;
  A = new int[I*J];
  B = new int[J*K];
  C = new int[I*K];

  const int max = 100;

  for (int i = 0; i != I*J; i++){
    A[i] = rand() % max;
  }
  for (int i = 0; i != J*K; i++){
    B[i] = rand() % max;
  }

  size_t Asize = I*J*sizeof(int);
  size_t Bsize = J*K*sizeof(int);
  size_t Csize = I*K*sizeof(int);

  cudaMalloc(&dA, Asize);
  cudaMalloc(&dB, Bsize);
  cudaMalloc(&dC, Csize);
  cudaCheckErrors("after cudaMalloc");

  cudaMemcpy(dA, A, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, Bsize, cudaMemcpyHostToDevice);
  cudaCheckErrors("after cudaMemcpy");

  dim3 block_size(block_side, block_side);
  dim3 nblocks(K / block_size.x + 1, I / block_size.y + 1);
  size_t bufsize = block_side * J * 2 * sizeof(int);
  printf("bufsize: %zu bytes\n", bufsize);
  matmul<<<nblocks, block_size, bufsize>>>(dA, dB, dC, block_side, I, J, K);
  cudaCheckErrors("after kernel call");

  cudaMemcpy(C, dC, Csize, cudaMemcpyDeviceToHost);
  cudaCheckErrors("after memcpy call");

  // check
  int check = 0;
  for (int i = 0; i != I; i++){
    for (int k = 0; k != K; k++){
      int dprod = 0;
      for (int j = 0; j != J; j++){
        dprod += A[i * J + j] * B[j * K + k];
      }
      if (dprod != C[i * K + k]) {
        check = 1;
        printf("Error C[%d][%d] = %d != %d (dprod)\n", i, k, C[i * K + k], dprod);
        goto END;
      }
    }
  }
END:
  if (! check)
    printf("success\n");

  cudaFree(dA);
  cudaFree(dB);

  delete[] A;
  delete[] B;
  delete[] C;

  return check;

}

