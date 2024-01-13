#include <stdio.h>

#define BLOCK_SIZE 256
#define RADIUS 3

__global__ void stencil_1d(int *in, int *out, int nout) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  // index into out
  int gindex = blockDim.x * blockIdx.x + threadIdx.x;

  if (gindex >= nout){
    return;
  }

  // center position of stencil in temp
  int lindex = RADIUS + threadIdx.x; 

  // populate temp
  temp[lindex] = in[gindex + RADIUS];
  
  // make the first three threads fill in the first three
  // elements of both halos
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex];
    temp[blockDim.x + threadIdx.x] = in[gindex + blockDim.x + RADIUS];
  }

  __syncthreads();

  int sum = 0;
  for (int i = -RADIUS; i != RADIUS; i++){
    sum += temp[lindex + i];
  }
  out[gindex] = sum;

}


int main(int argc, char **argv){
  srand(static_cast<unsigned>(time(0)));

  int nout = atoi(argv[1]);
  int nin = nout + 2 * RADIUS;

  int *in = new int[nin];
  int *out = new int[nout];

  for (int i=0; i != nin; i++){
    in[i] = rand() % 10;
  }

  int *g_in, *g_out;

  cudaMalloc(&g_in, nin * sizeof(int));
  cudaMalloc(&g_out, nout * sizeof(int));

  cudaMemcpy(g_in, in, nin * sizeof(int), cudaMemcpyHostToDevice);

  stencil_1d<<<nout / BLOCK_SIZE + 1, BLOCK_SIZE>>>(g_in, g_out, nout);

  cudaMemcpy(out, g_out, nout * sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d, %d, %d\n", in[0], in[1], in[2]);
  printf("%d, %d, %d\n", out[0], out[1], out[2]);

  delete[] in;
  delete[] out;
  cudaFree(g_in);
  cudaFree(g_out);
  
  return 0;
}
