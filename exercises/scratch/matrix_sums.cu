#include <stdio.h>
#include <math.h>

// error checking macro
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

const size_t DSIZE = 16384;      // matrix side dimension
const int block_size = 1024;  // CUDA maximum is 1024

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx * ds + i];         // write a for loop that will cause the thread to
        // iterate across a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}


__global__ void row_sums2(const float *A, float *sums, int row, size_t ds){
  __shared__ float bsums[1024];
  int tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + tid;
  bsums[tid] = 0.0f;

  // collect all data into the grid
  while (idx < ds){
    bsums[tid] += A[row * ds + idx];
    idx += gridDim.x * blockDim.x;
  }
  __syncthreads();

  // reduce a block to a single value
  for (int cs = 512; cs != 0; cs>>=1){
    if (tid < cs) 
      bsums[tid] += bsums[tid + cs];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(sums + row, bsums[0]);
}


__global__ void row_sums3(const float *A, float *sums, size_t ds){
  __shared__ float bsums[1024];
  int tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + tid;
  bsums[tid] = 0.0f;

  // collect all data into the grid
  while (idx < ds){
    bsums[tid] += A[blockIdx.y * ds + idx];
    idx += gridDim.x * blockDim.x;
  }
  __syncthreads();

  // reduce a block to a single value
  for (int cs = 512; cs != 0; cs>>=1){
    if (tid < cs) 
      bsums[tid] += bsums[tid + cs];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(sums + blockIdx.y, bsums[0]);
}


__global__ void row_sums4(const float *A, float *sums, size_t ds){
  __shared__ float bsums[32];
  int tid = threadIdx.x;
  int lid = tid % warpSize; 
  int wid = tid / warpSize;

  size_t idx = blockIdx.x * blockDim.x + tid;

  // collect all data into the grid
  float val = 0.0f;
  while (idx < ds){
    val = A[blockIdx.y * ds + idx];
    idx += gridDim.x * blockDim.x;
  }
  __syncthreads();

  unsigned mask = 0xFFFFFFFFU;

  // collect sums across each warp
  for (int offset = warpSize/2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(mask, val, offset);

  // if we're lane 0, store each warp summary in shared memory
  if (lid == 0)
    bsums[wid] = val;
  __syncthreads();


  // if we're warp 0, use each of the lanes to sum over shared memory
  // but, only use those lanes which have valid value.  this is
  // the number of warps in the block
  if (wid == 0) {
    val = (lid < blockDim.x/warpSize) ? bsums[lid] : 0;
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
      val += __shfl_down_sync(mask, val, offset);

    if (tid == 0) 
      atomicAdd(sums + blockIdx.y, val);
  }
}


__global__ void row_sums5(const float *A, float *sums, size_t ds, size_t nrowblocks){
  __shared__ float sdata[32];
  int tid = threadIdx.x;
  int lid = tid % warpSize;
  int wid = tid / warpSize;

  unsigned mask = 0xFFFFFFFFU;
  float val = 0.0f;
  int row_off = blockIdx.y * ds;
  int end = nrowblocks * blockDim.x;

  for (int idx = threadIdx.x; idx < end; idx += blockDim.x){
    val = idx < ds ? A[row_off + idx] : 0.0f;
    __syncthreads();
    for (int offset = warpSize/2; offset>0; offset>>=1)
      val += __shfl_down_sync(mask, val, offset);

    if (lid == 0)
      sdata[wid] = val;
    __syncthreads();

    if (wid == 0){
      val = tid < blockDim.x/warpSize ? sdata[lid] : 0.0f;
      for (int offset = warpSize/2; offset>0; offset>>=1)
        val += __shfl_down_sync(mask, val, offset);
    }
    // __syncthreads();
    if (tid == 0)
      sums[blockIdx.y] += val;
  }
}


// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds){

  int idx = blockIdx.x * blockDim.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[i * ds + idx];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}

bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}


bool validate_rowsums(const float *mat, const float *sums, size_t ds){
  for (int r = 0; r != ds; r++){
    float sum = 0.0f;
    size_t off = r * ds;
    for (int c = 0; c != ds; c++)
      sum += mat[off + c];
    if (fabs(sum - sums[r]) / sum > 1e-5){
      printf("row %d error: %f != %f\n", r, sum, sums[r]);
      return false;
    }
  }
  return true;
}




void post_call(const float *mat, float *d_sums, float *h_sums, const char msg[]){
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (validate_rowsums(mat, h_sums, DSIZE))
    printf("%s\n", msg);
}

int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
    
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 10.0 * (float)rand()/((float)RAND_MAX);
    
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float));
  // FIXME // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
    
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
    
  //cuda processing sequence step 1 is complete
  row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  post_call(h_A, d_sums, h_sums, "row_sums correct");

  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  for (int r = 0; r != DSIZE; r++)
    row_sums2<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, r, DSIZE);
  post_call(h_A, d_sums, h_sums, "row_sums2 correct");
    
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  dim3 grid((DSIZE+block_size-1)/block_size, DSIZE);
  dim3 block(block_size, 1);

  row_sums3<<<grid, block>>>(d_A, d_sums, DSIZE);
  post_call(h_A, d_sums, h_sums, "row_sums3 correct");
    
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  row_sums4<<<grid, block>>>(d_A, d_sums, DSIZE);
  post_call(h_A, d_sums, h_sums, "row_sums4 correct");
    
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  dim3 grid5(1, DSIZE);
  dim3 block5(block_size, 1);
  size_t nrow_blocks = DSIZE / block_size + int(DSIZE % block_size != 0);
  row_sums5<<<grid5, block5>>>(d_A, d_sums, DSIZE, nrow_blocks);
  post_call(h_A, d_sums, h_sums, "row_sums5 correct");
    
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
    
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  return 0;
}
  
