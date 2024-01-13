#include <stdio.h>
#include <stdlib.h>

__global__ void add(float *a, float *b, float *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

void fill_random(float *buf, size_t n) {
  for (auto i = 0; i < n; i++){
    buf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }
}


int main(int argc, char **argv) {
  srand(static_cast <unsigned> (time(0)));
  int n = atoi(argv[1]);
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  size_t dsize = n * sizeof(float);

  a = new float[dsize];
  b = new float[dsize];
  c = new float[dsize];

  fill_random(a, n);
  fill_random(b, n);

  cudaMalloc(&d_a, dsize);
  cudaMalloc(&d_b, dsize);
  cudaMalloc(&d_c, dsize);

  cudaMemcpy(d_a, a, dsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, dsize, cudaMemcpyHostToDevice);

  add<<<n, 1>>>(d_a, d_b, d_c, n);
  cudaMemcpy(c, d_c, dsize, cudaMemcpyDeviceToHost);
  printf("%f + %f = %f\n", a[0], b[0], c[0]);

}






