#include <stdio.h>

const int block_size = 32;

__global__ void matmul(const int *A, const int *B, int *C, int I, int J, int K) {
  __shared__ int arows[block_size][block_size];
  __shared__ int brows[block_size][block_size];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= I || x >= K) return;

  int dp = 0;
  for (int s = 0; s < J; s += block_size){
    int end = s + block_size < J ? s + block_size : J; 
    for (int j = s; j < end; j++){
      arows[threadIdx.y][j-s] = A[y * J + j];
      brows[j-s][threadIdx.x] = B[j * K + x];
    }
    __syncthreads();

    for (int j = s; j < end; j++)
      dp += arows[threadIdx.y][j-s] * brows[j-s][threadIdx.x];
  }
  C[y * K + x] = dp;
}


int main(int argc, char **argv){
  
  int I = atoi(argv[1]);
  int J = atoi(argv[2]);
  int K = atoi(argv[3]);

  int *A = new int[I*J];
  int *B = new int[J*K];
  int *C = new int[I*K];

  int *dA, *dB, *dC;

  size_t sizeA = I*J*sizeof(int);
  size_t sizeB = J*K*sizeof(int);
  size_t sizeC = I*K*sizeof(int);

  cudaMalloc(&dA, sizeA);
  cudaMalloc(&dB, sizeB);
  cudaMalloc(&dC, sizeC);

  const int max = 100;
  for (int i = 0; i != I*J; i++){
    A[i] = rand() % max;
  }
  for (int i = 0; i != J*K; i++){
    B[i] = rand() % max;
  }

  cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);
  
  dim3 grid(K / block_size + 1, I / block_size + 1);
  dim3 block(block_size, block_size);

  matmul<<<grid, block>>>(dA, dB, dC, I, J, K);

  cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

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
  cudaFree(dC);

  delete[] A;
  delete[] B;
  delete[] C;

  return check;
}




  

