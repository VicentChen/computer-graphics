#ifndef HITABLE_H__
#define HITABLE_H__

#include "Ray.h"

class Material;

struct HitRecord {
  float t;
  Vec3 p;
  Vec3 n;             // normal
  Material *mat_ptr;  // material
};

class Hitable {
 public:
  virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const = 0;
};

#endif  // !HITABLE_H__
