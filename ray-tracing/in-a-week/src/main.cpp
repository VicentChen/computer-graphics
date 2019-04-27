#include <iostream>
#include <svpng.inc>

inline void SaveImage(char* filepath, unsigned char* buffer, int width, int height, int channels);

int main() {
  //        width    height   channels
  const int W = 512, H = 512, C = 3;
  unsigned char img[W * H * C];
  
  const int R = 0, G = 1, B = 2, A = 3;
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      unsigned char* pixel = img + (i * W + j) * 3;
      pixel[R] = static_cast<unsigned char>(static_cast<float>(i) / H * 255);
      pixel[G] = static_cast<unsigned char>(static_cast<float>(j) / W * 255);
      pixel[B] = static_cast<unsigned char>((1 - static_cast<float>(i * j) / W / H) * 255);
    }
  }

  SaveImage("../../doc/img/in-a-week/test.png", img, W, H, C);
  return 0;
}

inline void SaveImage(char* filepath, unsigned char* buffer, int width, int height, int channels) {
  FILE* file = fopen(filepath, "wb");
  svpng(file, width, height, buffer, channels > 3);
  fclose(file);
}