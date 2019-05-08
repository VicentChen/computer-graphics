#ifndef COMMON_H__
#define COMMON_H__
#define M_PI 3.1415926536

#include <random>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifdef _WIN32
std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
float drand48() { return distribution(generator); }
#endif  // _WIN32

inline unsigned char* LoadImage(char* filepath, int* width, int* height, int* channels) {
  return stbi_load(filepath, width, height, channels, 0);
}

inline void FreeImage(unsigned char* img) { stbi_image_free(img); }

inline void SaveImage(char* filepath, unsigned char* buffer, int width, int height, int channels) {
  FILE* file = fopen(filepath, "wb");
  svpng(file, width, height, buffer, channels > 3);
  fclose(file);
}

#endif  // !COMMON_H__
