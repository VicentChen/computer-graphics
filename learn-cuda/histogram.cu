/**
 * TODO: find out the reason why shared memory is slower than global.
 */
 #include <iostream>
 #include <cmath>
 
 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <CImg.h>
 
 #include "common.h"
 
 #define COLOR_SIZE 256
 #define HISTOGRAM_WIDTH  (COLOR_SIZE << 2)
 #define HISTOGRAM_HEIGHT (COLOR_SIZE << 2)
 #define WARMUP true
 #define IMG_LEN 512
 
 using namespace std;
 using namespace cimg_library;
 
 __shared__ unsigned int shared_buffer[COLOR_SIZE];
 
 bool check_img(CImg<unsigned char>& H);
 void warmup(unsigned char* gpu_img, int *device_buffer);
 __global__ void gpu_histogram_atomicadd(unsigned char *I, int *H);
 __global__ void gpu_histogram_atomicadd_4B(unsigned char *I, int *H);
 __global__ void gpu_histogram_atomicadd_4B_shared(unsigned char *I, int *H);
 __global__ void gpu_histogram_atomicadd_4B_half_shared(unsigned char *I, int *H);
 __global__ void gpu_histogram_atomicadd_4B_quarter_shared(unsigned char *I, int *H);
 __global__ void gpu_histogram_atomicadd_16B_shared(unsigned char *I, int *H);
 bool check_consistency(int *cpu_buffer, int *gpu_buffer);
 void draw_histogram(int *B, CImg<unsigned char>& H, char* filename);
 
 int main(int argc, char* argv[]) {
   // ----- time measurement ----- //
 #ifdef __linux__
   double start, end;
 #elif _WIN32
   unsigned long long start, end;
 #endif
 
   // ----- load image and get image characteristics ----- //
   CImg<unsigned char> img("img/husky.jpg"), histogram(HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, 1, 1, 255);
   CImg<unsigned char> gray_img = img.RGBtoYCbCr().channel(0); // convert to grayscale
   if (!check_img(gray_img)) return -1;
   int picture_size = IMG_LEN * IMG_LEN;
 
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
   CHECK(cudaMalloc((unsigned char**)&gpu_img, picture_size)); // allocate space for image
   CHECK(cudaMalloc((int**)&device_buffer, sizeof(gpu_buffer))); // allocate space for histogram
   CHECK(cudaMemcpy(gpu_img, img.data(), picture_size, cudaMemcpyHostToDevice));
   CHECK(cudaMemset(device_buffer, 0, sizeof(gpu_buffer)));
   CHECK(cudaDeviceSynchronize());
 
   // ----- warm up ----- //
   if (WARMUP) warmup(gpu_img, device_buffer);
 
   //       ----- gpu compute histogram with atomicAdd -----       //
   cout << "----- gpu compute histogram with atomicAdd -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   gpu_histogram_atomicadd<<<IMG_LEN, IMG_LEN>>>(gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   //       ----- gpu compute histogram with atomicAdd(4 Bytes/Thread) -----       //
   cout << "----- gpu compute histogram with atomicAdd(4 Bytes/Thread) -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   gpu_histogram_atomicadd_4B<<<IMG_LEN, IMG_LEN / 4>>> (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   //       ----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using shared memory -----       //
   cout << "----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using shared memory -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   // set 256 threads / block to fit shared_buffer
   gpu_histogram_atomicadd_4B_shared<<<picture_size / COLOR_SIZE / 4, COLOR_SIZE>>> (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   //       ----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using half shared memory -----       //
   cout << "----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using half shared memory -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   // set 128 threads / block to reduce shared memory usage
   gpu_histogram_atomicadd_4B_half_shared<<<picture_size / COLOR_SIZE / 2, COLOR_SIZE / 2>>> (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   //       ----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using quarter shared memory -----       //
   cout << "----- gpu compute histogram with atomicAdd(4 Bytes/Thread) using quarter shared memory -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   // set 128 threads / block to reduce shared memory usage
   gpu_histogram_atomicadd_4B_quarter_shared<<<picture_size / COLOR_SIZE, COLOR_SIZE / 4>>> (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   //       ----- gpu compute histogram with atomicAdd(16 Bytes/Thread) using shared memory -----       //
   cout << "----- gpu compute histogram with atomicAdd(16 Bytes/Thread) using shared memory -----" << endl;
   CHECK(cudaMemset(device_buffer, 0, COLOR_SIZE * sizeof(int))); // clear buffer
   CHECK(cudaDeviceSynchronize());
 
   start = seconds();
   // set 256 threads / block to fit shared_buffer
   gpu_histogram_atomicadd_16B_shared <<<picture_size / COLOR_SIZE / 16, COLOR_SIZE>>> (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
   end = seconds();
   cout << "TIME: " << end - start << " s" << endl;
 
   // copy histogram back
   CHECK(cudaMemcpy(gpu_buffer, device_buffer, sizeof(gpu_buffer), cudaMemcpyDeviceToHost));
   CHECK(cudaDeviceSynchronize());
 
   check_consistency(cpu_buffer, gpu_buffer);
 
   // ----- device release ----- //
   CHECK(cudaFree(gpu_img));
   CHECK(cudaFree(device_buffer));
   CHECK(cudaDeviceReset());
 
   // ----- save histogram ----- //
   draw_histogram(cpu_buffer, histogram, "img/husky-histogram-cpu.jpg");
   draw_histogram(gpu_buffer, histogram, "img/husky-histogram-gpu.jpg");
 
   return 0;
 }
 
 bool check_img(CImg<unsigned char>& H) {
 #ifndef IMG_LEN
   cout << "ERROR: marco IMG_LEN not defined" << endl;
   return false;
 #else
   if (H.width() != IMG_LEN || H.height() != IMG_LEN) {
     cout << "ERROR: image size( " << H.width() << " * " << H.height() << " ) NOT MATCH marco ( IMG_LEN * IMG_LEN ), IMG_LEN = " << IMG_LEN << endl;
     return false;
   }
   if (H.spectrum() != 1) {
     cout << "ERROR: image should have only 1 channel, now has " << H.spectrum() << " channels." << endl;
     return false;
   }
   return true;
 #endif
 }
 
 void warmup(unsigned char* gpu_img, int *device_buffer) {
   gpu_histogram_atomicadd<<<1, 1>>> (gpu_img, device_buffer);
   gpu_histogram_atomicadd_4B<<<1, 1>>> (gpu_img, device_buffer);
   gpu_histogram_atomicadd_4B_shared<<<1, 1>>> (gpu_img, device_buffer);
   gpu_histogram_atomicadd_4B_half_shared<<<1, 1>>> (gpu_img, device_buffer);
   gpu_histogram_atomicadd_4B_quarter_shared<<<1, 1>>> (gpu_img, device_buffer);
   gpu_histogram_atomicadd_16B_shared << <1, 1 >> > (gpu_img, device_buffer);
   CHECK(cudaDeviceSynchronize());
 }
 
 __global__ void gpu_histogram_atomicadd(unsigned char *I, int *H) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   atomicAdd(&H[I[idx]], 1);
 }
 
 __global__ void gpu_histogram_atomicadd_4B(unsigned char *I, int *H) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int value = ((const unsigned int*)I)[idx];
 
   atomicAdd(&H[(value & 0x000000FF) >>  0], 1);
   atomicAdd(&H[(value & 0x0000FF00) >>  8], 1);
   atomicAdd(&H[(value & 0x00FF0000) >> 16], 1);
   atomicAdd(&H[(value & 0xFF000000) >> 24], 1);
 }
 
 __global__ void gpu_histogram_atomicadd_4B_shared(unsigned char *I, int *H) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int value = ((const unsigned int*)I)[idx];
 
   shared_buffer[threadIdx.x] = 0; // clear shared memory
   __syncthreads();
 
   atomicAdd(&shared_buffer[(value & 0x000000FF) >>  0], 1);
   atomicAdd(&shared_buffer[(value & 0x0000FF00) >>  8], 1);
   atomicAdd(&shared_buffer[(value & 0x00FF0000) >> 16], 1);
   atomicAdd(&shared_buffer[(value & 0xFF000000) >> 24], 1);
   __syncthreads();
 
   atomicAdd(&H[threadIdx.x], shared_buffer[threadIdx.x]);
 }
 
 __global__ void gpu_histogram_atomicadd_4B_half_shared(unsigned char *I, int *H) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int value = ((const unsigned int*)I)[idx];
 
   shared_buffer[(threadIdx.x << 1)    ] = 0; // clear shared memory
   shared_buffer[(threadIdx.x << 1) + 1] = 0; // clear shared memory
   __syncthreads();
 
   atomicAdd(&shared_buffer[(value & 0x000000FF) >>  0], 1);
   atomicAdd(&shared_buffer[(value & 0x0000FF00) >>  8], 1);
   atomicAdd(&shared_buffer[(value & 0x00FF0000) >> 16], 1);
   atomicAdd(&shared_buffer[(value & 0xFF000000) >> 24], 1);
   __syncthreads();
 
   atomicAdd(&H[(threadIdx.x << 1)    ], shared_buffer[(threadIdx.x << 1)    ]);
   atomicAdd(&H[(threadIdx.x << 1) + 1], shared_buffer[(threadIdx.x << 1) + 1]);
 }
 
 __global__ void gpu_histogram_atomicadd_4B_quarter_shared(unsigned char *I, int *H) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned int value = ((const unsigned int*)I)[idx];
 
   shared_buffer[(threadIdx.x << 2)    ] = 0; // clear shared memory
   shared_buffer[(threadIdx.x << 2) + 1] = 0; // clear shared memory
   shared_buffer[(threadIdx.x << 2) + 2] = 0; // clear shared memory
   shared_buffer[(threadIdx.x << 2) + 3] = 0; // clear shared memory
   __syncthreads();
 
   atomicAdd(&shared_buffer[(value & 0x000000FF) >>  0], 1);
   atomicAdd(&shared_buffer[(value & 0x0000FF00) >>  8], 1);
   atomicAdd(&shared_buffer[(value & 0x00FF0000) >> 16], 1);
   atomicAdd(&shared_buffer[(value & 0xFF000000) >> 24], 1);
   __syncthreads();
 
   atomicAdd(&H[(threadIdx.x << 2)    ], shared_buffer[(threadIdx.x << 2)    ]);
   atomicAdd(&H[(threadIdx.x << 2) + 1], shared_buffer[(threadIdx.x << 2) + 1]);
   atomicAdd(&H[(threadIdx.x << 2) + 2], shared_buffer[(threadIdx.x << 2) + 2]);
   atomicAdd(&H[(threadIdx.x << 2) + 3], shared_buffer[(threadIdx.x << 2) + 3]);
 }
 
 __global__ void gpu_histogram_atomicadd_16B_shared(unsigned char *I, int *H) {
   int idx = (blockIdx.x * blockDim.x + threadIdx.x) << 2; // 4 ints (16 chars) per thread
   const unsigned int *value = (const unsigned int*)I + idx;
 
   shared_buffer[threadIdx.x] = 0; // clear shared memory
   __syncthreads();
 
   for (int i = 0; i < 4; ++i) {
     atomicAdd(&shared_buffer[(value[i] & 0x000000FF) >>  0], 1);
     atomicAdd(&shared_buffer[(value[i] & 0x0000FF00) >>  8], 1);
     atomicAdd(&shared_buffer[(value[i] & 0x00FF0000) >> 16], 1);
     atomicAdd(&shared_buffer[(value[i] & 0xFF000000) >> 24], 1);
   }
   __syncthreads();
 
   atomicAdd(&H[threadIdx.x], shared_buffer[threadIdx.x]);
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