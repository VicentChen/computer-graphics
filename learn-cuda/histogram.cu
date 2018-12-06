#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CImg.h>

#include "common.h"

#define COLOR_SIZE 256
#define HISTOGRAM_WIDTH  (COLOR_SIZE)
#define HISTOGRAM_HEIGHT (COLOR_SIZE)

using namespace std;
using namespace cimg_library;

__global__ void gpu_compute_histogram(unsigned char *I, int *H);
__host__ void draw_histogram(int *B, CImg<unsigned char>& H, char* filename);

int main(int argc, char* argv[]) {
  CImg<unsigned char> img("husky.jpg"), histogram(HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, 1, 1, 255);
  CImg<unsigned char> gray_img = img.RGBtoYCbCr().channel(0); // convert to grayscale

  int picture_size = gray_img.size() * sizeof(unsigned char) * gray_img.spectrum();

  // ----- gpu compute histogram ----- //
  int gpu_buffer[COLOR_SIZE];
  CHECK(cudaSetDevice(0)); // init cuda
  memset(gpu_buffer, 0, sizeof(gpu_buffer)); // init buffer

  // copy to device
  unsigned char *gpu_img;
  int *device_buffer;
  CHECK(cudaMalloc((unsigned char**)&gpu_img, picture_size));
  CHECK(cudaMalloc((int**)&device_buffer, sizeof(gpu_buffer)));
  CHECK(cudaMemcpy(gpu_img, img.data(), picture_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device_buffer, gpu_buffer, sizeof(gpu_buffer), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());

  // TODO: warm up
  gpu_compute_histogram<<<728, 728>>>(gpu_img, device_buffer);
  CHECK(cudaDeviceSynchronize());

  // copy histogram back
  CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());

  // draw gpu histogram
  draw_histogram(gpu_buffer, histogram, "husky-histogram-gpu.jpg");

  // free device
  CHECK(cudaFree(gpu_img));
  CHECK(cudaFree(device_buffer));
  CHECK(cudaDeviceReset());

  // ----- cpu compute histogram ----- //
  int cpu_buffer[COLOR_SIZE];
  memset(cpu_buffer, 0, sizeof(cpu_buffer));
  cimg_for(gray_img, p, unsigned char) { cpu_buffer[*p]++; }
  draw_histogram(cpu_buffer, histogram, "husky-histogram-cpu.jpg");

  // ----- check consistent ----- //
  for (int i = 0; i < COLOR_SIZE; i++)
    if (cpu_buffer[i] != gpu_buffer[i])
      cout << i << ": " << cpu_buffer[i] << '\t' << gpu_buffer[i] << endl;

  return 0;
}

__global__ void gpu_compute_histogram(unsigned char *I, int *H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&H[I[idx]], 1);
}

__host__ void draw_histogram(int *B, CImg<unsigned char>& H, char* filename) {
  int maxCount = 0;
  for (int i = 0; i < COLOR_SIZE; ++i)
    if (B[i] > maxCount) maxCount = B[i];
  const int heightScale = (int)ceil(maxCount * 1.0 / HISTOGRAM_WIDTH);
  const int widthScale = (int)HISTOGRAM_HEIGHT / COLOR_SIZE;
  const unsigned char color[] = { 128 };

  H.fill(255); // clear canvas
  for (int i = 0; i < COLOR_SIZE; ++i) {
    H.draw_rectangle(i * widthScale, HISTOGRAM_HEIGHT, i * widthScale + widthScale - 1, HISTOGRAM_HEIGHT - B[i] / heightScale, color);
  }

  H.save_jpeg(filename);
}