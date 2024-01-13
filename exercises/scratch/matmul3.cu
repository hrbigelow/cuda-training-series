/*
Approach:
one thread per output element in the output matrix
a threadblock will be organized as a 2D square region of the output matrix
the grid will be a 2D region of threadblocks covering the output matrix

The thread computation will be a block strided loop through the inner dimension J.
each iteration will load a square region of A and of B into shared memory, then
synchronize.

Then, each thread will accumulate its dot product from this shared memory


   */

#include <stdio.h>

const int side = 32;

__global__ void mmul(const int *A, const int *B, int *C, int Y, int D, int X){
  // global x dimension
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ int a[side][side];
  __shared__ int b[side][side];

  int dp = 0;

  for (int d = 0; d < D; d += side){
    int doff1 = d + threadIdx.x;
    int doff2 = d + threadIdx.y;
    a[threadIdx.y][threadIdx.x] = doff1 < D ? A[y * D + doff1] : 0.0;
    b[threadIdx.y][threadIdx.x] = doff2 < D ? B[doff2 * X + x] : 0.0;
    __syncthreads();

    for (int o = 0; (o != side) && ((o + d) < D); o++)
      dp += a[threadIdx.y][o] * b[o][threadIdx.x];
    __syncthreads();

  }
  if (x < X && y < Y)
    C[y * X + x] = dp;
}

void check(const int *A, const int *B, const int *C, int Y, int D, int X){
  for (int y = 0; y != Y; y++){
    for (int x = 0; x != X; x++){
      int dp = 0;
      for (int d = 0; d != D; d++){
        dp += A[y * D + d] * B[d * X + x];
      }
      if (dp != C[y * X + x]) {
        printf("C[%d][%d] != dp:  %d != %d\n", y, x, C[y * X + x], dp);
      }
    }
  }
}


int main(int argc, char **argv){
  int Y = atoi(argv[1]);
  int X = atoi(argv[2]);
  int D = atoi(argv[3]);

  int *A, *B, *C, *dA, *dB, *dC;
  size_t sizeA = Y*D*sizeof(int);
  size_t sizeB = D*X*sizeof(int);
  size_t sizeC = Y*X*sizeof(int);
  A = (int *)malloc(sizeA);
  B = (int *)malloc(sizeB);
  C = (int *)malloc(sizeC);

  cudaMalloc(&dA, sizeA);
  cudaMalloc(&dB, sizeB);
  cudaMalloc(&dC, sizeC);

  // init
  int max = 1000;
  for (int i = 0; i != Y*D; i++)
    A[i] = rand() % max;

  for (int i = 0; i != D*X; i++)
    B[i] = rand() % max;

  // copy
  cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

  dim3 grid(X / side + 1, Y / side + 1);
  dim3 block(side, side);

  mmul<<<grid, block>>>(dA, dB, dC, Y, D, X);
  cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

  check(A, B, C, Y, D, X);
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}


