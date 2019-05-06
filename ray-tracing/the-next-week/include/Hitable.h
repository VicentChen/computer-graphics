#ifndef HITABLE_H__
#define HITABLE_H__

#include "Ray.h"
#include "AABB.h"

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
  virtual bool bounding_box(float t0, float t1, AABB &box) const = 0;
};

#endif  // !HITABLE_H__
