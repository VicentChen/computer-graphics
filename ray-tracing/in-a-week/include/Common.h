#ifndef COMMON_H__
#define COMMON_H__
#define M_PI 3.1415926536

#include <random>

#ifdef _WIN32
std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
float drand48() { return distribution(generator); }
#endif  // _WIN32

inline void SaveImage(char* filepath, unsigned char* buffer, int width, int height, int channels) {
  FILE* file = fopen(filepath, "wb");
  svpng(file, width, height, buffer, channels > 3);
  fclose(file);
}

#endif  // !COMMON_H__
