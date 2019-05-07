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

#endif  // !TEXTURE_H__
