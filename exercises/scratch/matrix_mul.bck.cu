#include <stdio.h>

__global__ void matmul(const float *a, const float *b, float *c, size_t i, size_t j,
    size_t k) {
  int ai = threadIdx.x + blockIdx.x * threadDim.x;
  int bi = blockIdx.y + threadIdx.x * blockDim.y;
  int ci = blockIdx.y + blockIdx.x * blockDim.y;

  c[ci] += a[ai] * b[bi];
}

int main() {

  size_t X = 10;
  size_t Y = 20;
  size_t J = 100;

  float *a, *b, *c, *d_a, *d_b, *d_c;
  size_t asize = X * J * sizeof(float);
  size_t bsize = J * Y * sizeof(float);
  size_t csize = X * Y * sizeof(float);

  a = new float[asize];
  b = new float[bsize];
  c = new float[csize];

  cudaMalloc(&d_a, asize);
  cudaMalloc(&d_b, bsize);
  cudaMalloc(&d_c, csize);

  cudaMemcpy(d_a, a, asize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bsize, cudaMemcpyHostToDevice);


}



