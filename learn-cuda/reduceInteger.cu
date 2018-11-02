#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

double seconds();
int cpuReduce(int *N, int const size);
__global__ void warmup(int *I, int *O, unsigned int *N);
__global__ void gpuReduceRecursive(int *I, int *O, unsigned int n);
__global__ void gpuReduceRecursiveL(int *I, int *O, unsigned int n);
__global__ void gpuReduceInterleaved(int *I, int *O, unsigned int n);
__global__ void gpuReduceInterleavedUnrolling2(int *I, int *O, unsigned int n);

int main(int argc, char **argv) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  CHECK(cudaSetDevice(dev));
  printf("%s starting... reduction\n", argv[0]);
  printf("Using device %d: %s\n", dev, deviceProp.name);

  // initialization 
  int size = 1 << 14;
  printf("With array size %d\n", size);

  // execution configuration
  int blockSize = 512;
  if (argc > 1) blockSize = atoi(argv[1]);
  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("grid(%d,1), block(%d,1)\n", grid.x, block.x);

  // allocate host memory
  size_t bytes = size * sizeof(int);
  int *ipt = (int*)malloc(bytes);
  int *opt = (int*)malloc(grid.x * sizeof(int));
  int *tmp = (int*)malloc(bytes);

  // allocate device memory
  int *d_I, *d_O;
  CHECK(cudaMalloc((int**)&d_I, bytes));
  CHECK(cudaMalloc((int**)&d_O, grid.x * sizeof(int)));

  // initialize host array
  for (int i = 0; i < size; i++)
    ipt[i] = (int)(rand() & 0xFF);
  memcpy(tmp, ipt, bytes);

  double iStart, iElaps;
  int cpuSum, gpuSum;

  // ---------- CPU reduce ---------- //
  iStart = seconds();
  cpuSum = cpuReduce(tmp, size);
  iElaps = seconds() - iStart;
  printf("CPU: %lfms\n", iElaps);
  
  // ---------- KERNEL 1: Original Reduce ---------- //
  CHECK(cudaMemcpy(d_I, ipt, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = seconds();
  gpuReduceRecursive<<<grid, block>>>(d_I, d_O, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = seconds() - iStart;
  printf("GPU(KERNEL 1): %lfms\n", iElaps);
  
  CHECK(cudaMemcpy(opt, d_O, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
  gpuSum = 0;
  for(int i = 0; i < grid.x; i++)
    gpuSum += opt[i];

  if (gpuSum != cpuSum) 
    printf("Kernel 1 does not match.\nCPU: %d GPU %d\n", cpuSum, gpuSum);
  
  // ---------- KERNEL 2: Original Reduce L ---------- //
  CHECK(cudaMemcpy(d_I, ipt, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = seconds();
  gpuReduceRecursiveL<<<grid, block>>>(d_I, d_O, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = seconds() - iStart;
  printf("GPU(KERNEL 2): %lfms\n", iElaps);

  CHECK(cudaMemcpy(opt, d_O, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
  gpuSum = 0;
  for(int i = 0; i < grid.x; i++)
    gpuSum += opt[i];

  if (gpuSum != cpuSum) 
    printf("Kernel 2 does not match.\nCPU: %d GPU %d\n", cpuSum, gpuSum);

  // ---------- KERNEL 3: Original Reduce ---------- //
  CHECK(cudaMemcpy(d_I, ipt, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = seconds();
  gpuReduceInterleaved<<<grid, block>>>(d_I, d_O, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = seconds() - iStart;
  printf("GPU(KERNEL 3): %lfms\n", iElaps);

  CHECK(cudaMemcpy(opt, d_O, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
  gpuSum = 0;
  for(int i = 0; i < grid.x; i++)
    gpuSum += opt[i];

  if (gpuSum != cpuSum) 
    printf("Kernel 3 does not match.\nCPU: %d GPU %d\n", cpuSum, gpuSum);

  // ---------- KERNEL 4: Original Reduce ---------- //
  CHECK(cudaMemcpy(d_I, ipt, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  iStart = seconds();
  gpuReduceInterleavedUnrolling2<<<grid.x / 2, block>>>(d_I, d_O, size);
  CHECK(cudaDeviceSynchronize());
  iElaps = seconds() - iStart;
  printf("GPU(KERNEL 4): %lfms\n", iElaps);

  CHECK(cudaMemcpy(opt, d_O, grid.x * sizeof(int) / 2, cudaMemcpyDeviceToHost));
  gpuSum = 0;
  for(int i = 0; i < grid.x / 2; i++)
    gpuSum += opt[i];

  if (gpuSum != cpuSum) 
    printf("Kernel 4 does not match.\nCPU: %d GPU %d\n", cpuSum, gpuSum);

  // ---------- KERNEL 4: Original Reduce ---------- //
  // TODO: copy host data to device
  // TODO: reduce on device
  // TODO: copy device result to host
  // TODO: print gpu time

  // TODO: compute gpu sum
  // TODO: check gpu sum

  free(ipt);
  free(opt);
  free(tmp);

  CHECK(cudaFree(d_I));
  CHECK(cudaFree(d_O));

  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}

int cpuReduce(int *N, const int size) {
  if (size == 1) return N[0];

  int stride = size / 2;
  for(int i = 0; i < stride; i++)
    N[i] += N[i + stride];

  return cpuReduce(N, stride);
}

/*
  EXAMPLE:

    Loop 1:
      stride  :<_>_ _ _ _ _ _ _ 
      block   :|_|_|_|_|_|_|_|_|
      thread  : 0 1 2 3 4 5 6 7
                | / | / | / | /
                |/  |/  |/  |/
      inactive: | 1 | 3 | 5 | 7 ===> (tid % (2 * 1) != 0)
      active  : 0   2   4   6   ===> (tid % (2 * 1) == 0)
    
    Loop 2:
      stride  :<_ _>_ _ _ _ _ _ 
      block   :|_|_|_|_|_|_|_|_|
      thread  : 0   2   4   6
                |  /    |  /
                | /     | /
      inactive: |/  2   |/  6   ===> (tid % (2 * 2) != 0)
      active  : 0       4       ===> (tid % (2 * 2) == 0)
    
    Loop 3:
      stride  :<_ _ _ _>_ _ _ _ 
      block   :|_|_|_|_|_|_|_|_|
      thread  : 0       4      
                |      /
                |    /
                |  /
                |/
      inactive: |       4       ===> (tid % (2 * 4) != 0)
      active  : 0               ===> (tid % (2 * 4) == 0)
  
  ANALYZE:

    Only about half threads will be executed.
*/
__global__ void gpuReduceRecursive(int *I, int *O, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  int *N = I + blockIdx.x * blockDim.x; // N: begin address of global memory in block
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0)
      N[tid] += N[tid + stride];

    __syncthreads();
  }

  if (tid == 0) O[blockIdx.x] = N[0];
}

/*
  EXAMPLE
              _ _ _ _ _ _ _ _ 
    block   :|_|_|_|_|_|_|_|_|
    Loop1   : 0 | 1 | 2 | 3 |  4...7(inactive)
              |/  |/  |/  |/   
    Loop2   : 0   |   1   |    2...7(inactive)
              |  /    |  /
              | /     | /
              |/      |/
    Loop3   : 0_______|        1...7(inactive)
              |
              0
  
  ANALYZE
    
    A bunch of thread will active, another will inactive.
*/
__global__ void gpuReduceRecursiveL(int *I, int *O, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  int *N = I + blockIdx.x * blockDim.x; // N: begin address of global memory in block
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x)
      N[index] += N[index + stride];
    
    __syncthreads();
  }

  if (tid == 0) O[blockIdx.x] = N[0];
}

/*
  EXAMPLE
              _ _ _ _ _ _ _ _ 
    block   :|_|_|_|_|_|_|_|_|
    Loop1   : 0 1 2 3(4 5 6 7)
              | | | |/ / / /
              | | |/|/ / /
              | |/|/|/ /
              |/|/|/|/
    Loop2   : 0 1(2 3)
              | |/ /
              |/|/
    Loop3   : 0(1)
              |/
              0
    *: threads in () are inactive. 

  ANALYZE
    
    More efficient memory usage.
*/
__global__ void gpuReduceInterleaved(int *I, int *O, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  int *N = I + blockIdx.x * blockDim.x; // N: begin address of global memory in block
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      N[tid] += N[tid + stride];

    __syncthreads();
  }

  if (tid == 0) O[blockIdx.x] = N[0];
}

/*
  EXAMPLE:
              _ _ _ _ _ _ _ _ 
    block   :|_|_|_|_|_|_|_|_|
              0 1 2 3 4 5 6 7 
    unroll  : | | | |/ / / /
              | | |/|/ / /
              | |/|/|/ /
              |/|/|/|/
              0 1 2 3
    gpuReduceInterleaved();
*/
__global__ void gpuReduceInterleavedUnrolling2(int *I, int *O, unsigned int n){
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + 2 * blockIdx.x * blockDim.x;

  if (idx + blockDim.x < n) I[idx] += I[idx + blockDim.x]; // unroll
  __syncthreads();

  int *N = I + 2 * blockIdx.x * blockDim.x;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) N[tid] += N[tid + stride];
    __syncthreads();
  }
  
  if (tid == 0) O[blockIdx.x] = N[0];
}
