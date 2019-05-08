#ifndef TEXTURE_H__
#define TEXTURE_H__

#include <cmath>
#include "Perlin.h"
#include "Vec3.h"

class Texture {
 public:
  virtual Vec3 value(float u, float v, const Vec3& p) const = 0;
};

class ConstantTexture : public Texture {
 public:
  ConstantTexture() {}
  ConstantTexture(Vec3 c) : color(c) {}

  virtual Vec3 value(float u, float v, const Vec3& p) const { return color; }

  Vec3 color;
};

class CheckerTexture : public Texture {
 public:
  CheckerTexture() {}
  CheckerTexture(Texture* t0, Texture* t1) : even(t0), odd(t1) {}

  virtual Vec3 value(float u, float v, const Vec3& p) const {
    float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
    if (sines < 0)
      return odd->value(u, v, p);
    else
      return even->value(u, v, p);
  }

  Texture *odd, *even;
};

class NoiseTexture : public Texture {
 public:
  NoiseTexture() {}
  NoiseTexture(float scale_) : scale(scale_) {}
  virtual Vec3 value(float u, float v, const Vec3& p) const {
    return Vec3(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
  }
  Perlin noise;
  float scale;
};

class ImageTexture : public Texture{
 public:
  ImageTexture() {}
  ImageTexture(unsigned char* data_, int W_, int H_) : data(data_), W(W_), H(H_) {}

  virtual Vec3 value(float u, float v, const Vec3& p) const {
    int i = u * W;
    int j = (1 - v) * H - 0.001;
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > W - 1) i = W - 1;
    if (j > H - 1) j = H - 1;
    float r = int(data[3 * i + 3 * W * j + 0]) / 255.0;
    float g = int(data[3 * i + 3 * W * j + 1]) / 255.0;
    float b = int(data[3 * i + 3 * W * j + 2]) / 255.0;
    return Vec3(r, g, b);
  }

  unsigned char *data;
  int W, H;
};



#endif  // !TEXTURE_H__
