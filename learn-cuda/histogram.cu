#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CImg.h>

#include "common.h"

#define COLOR_SIZE 256
#define HISTOGRAM_WIDTH  (COLOR_SIZE << 2)
#define HISTOGRAM_HEIGHT (COLOR_SIZE << 2)

using namespace std;
using namespace cimg_library;

void warmup(unsigned char* gpu_img, int *device_buffer);
__global__ void gpu_histogram_atomicadd(unsigned char *I, int *H);
bool check_consistency(int *cpu_buffer, int *gpu_buffer);
void draw_histogram(int *B, CImg<unsigned char>& H, char* filename);

int main(int argc, char* argv[]) {
  // ----- time measurement ----- //
  double start, end;

  // ----- load image and get image characteristics ----- //
  CImg<unsigned char> img("img/husky.jpg"), histogram(HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, 1, 1, 255);
  CImg<unsigned char> gray_img = img.RGBtoYCbCr().channel(0); // convert to grayscale
  int picture_size = gray_img.size() * sizeof(unsigned char) * gray_img.spectrum();

  //       ----- cpu compute histogram-----       //
  cout << "----- cpu compute histogram -----" << endl;
  int cpu_buffer[COLOR_SIZE];
  memset(cpu_buffer, 0, sizeof(cpu_buffer));
  start = seconds();
  cimg_for(gray_img, p, unsigned char) { cpu_buffer[*p]++; }
  end = seconds();
  cout << "TIME: " << end - start << " s" << endl;

  // ----- gpu preparation ----- //
  int gpu_buffer[COLOR_SIZE]; // buffer for 
  unsigned char *gpu_img; // device memory for image
  int *device_buffer; // device buffer for histogram
  CHECK(cudaSetDevice(0)); // init cuda
  memset(gpu_buffer, 0, sizeof(gpu_buffer)); // init buffer
  CHECK(cudaMalloc((unsigned char**)&gpu_img, picture_size)); // allocate space for image
  CHECK(cudaMalloc((int**)&device_buffer, sizeof(gpu_buffer))); // allocate space for histogram
  CHECK(cudaMemcpy(gpu_img, img.data(), picture_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device_buffer, gpu_buffer, sizeof(gpu_buffer), cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());

  // ----- warm up ----- //
  //warmup(gpu_img, device_buffer);

  //       ----- gpu compute histogram with atomicAdd() -----       //
  cout << "----- gpu compute histogram with atomicAdd() -----" << endl;
  start = seconds();
  gpu_histogram_atomicadd<<<728, 728>>>(gpu_img, device_buffer);
  CHECK(cudaDeviceSynchronize());
  end = seconds();
  cout << "TIME: " << end - start << " s" << endl;

  // copy histogram back
  CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());

  if (!check_consistency(cpu_buffer, gpu_buffer))
    return 0;

  // ----- device release ----- //
  CHECK(cudaFree(gpu_img));
  CHECK(cudaFree(device_buffer));
  CHECK(cudaDeviceReset());

  // ----- save histogram ----- //
  draw_histogram(cpu_buffer, histogram, "img/husky-histogram-cpu.jpg");
  draw_histogram(gpu_buffer, histogram, "img/husky-histogram-gpu.jpg");

  return 0;
}

void warmup(unsigned char* gpu_img, int *device_buffer) {
  gpu_histogram_atomicadd <<<1, 1>>> (gpu_img, device_buffer);
  CHECK(cudaDeviceSynchronize());
}

__global__ void gpu_histogram_atomicadd(unsigned char *I, int *H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&H[I[idx]], 1);
}

void draw_histogram(int *B, CImg<unsigned char>& H, char* filename) {
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

bool check_consistency(int *cpu_buffer, int *gpu_buffer) {
  bool is_consistent = true;
  for (int i = 0; i < COLOR_SIZE; i++)
    if (cpu_buffer[i] != gpu_buffer[i]) {
      cout << i << ": " << cpu_buffer[i] << '\t' << gpu_buffer[i] << endl;
      is_consistent = false;
    }
  if (!is_consistent)
    cout << "ERROR: Not consistent" << endl;
  return is_consistent;
}