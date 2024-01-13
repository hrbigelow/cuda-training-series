#include <stdio.h>

__global__ void vector_add(const int *a, const int *b, int *c, int nelem){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nelem){
    c[idx] = a[idx] + b[idx];
  }
}

const int NELEM = 100000000;
const int BLOCK_SIZE = 1024;
const int NBLOCKS = NELEM / BLOCK_SIZE + 1;
const int VAL_MAX = 1000;

int main(){
  int *a, *b, *c, *da, *db, *dc;
  a = new int[NELEM];
  b = new int[NELEM];
  c = new int[NELEM];

  for (int i = 0; i != NELEM; i++){
    a[i] = rand() % VAL_MAX; 
    b[i] = rand() % VAL_MAX;
  }

  size_t dsize = NELEM * sizeof(int);

  cudaMalloc(&da, dsize);
  cudaMalloc(&db, dsize);
  cudaMalloc(&dc, dsize);

  cudaMemcpy(da, a, dsize, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, dsize, cudaMemcpyHostToDevice);

  vector_add<<<NBLOCKS, BLOCK_SIZE>>>(da, db, dc, NELEM);
  // vector_add<<<NELEM, 1>>>(da, db, dc, NELEM);
  cudaMemcpy(c, dc, dsize, cudaMemcpyDeviceToHost);

  // test
  for (int i = 0; i != NELEM; i++){
    if (c[i] != a[i] + b[i]) {
      printf("Failed at index %d: %d != %d + %d\n", i, c[i], a[i], b[i]);
      return 1;
    }
  }
  printf("succeeded.\n");
  return 0;

  delete[] a;
  delete[] b;
  delete[] c;
}


