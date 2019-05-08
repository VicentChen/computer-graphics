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
  float u;
  float v;
};

class Hitable {
 public:
  virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const = 0;
  virtual bool bounding_box(float t0, float t1, AABB &box) const = 0;
};

class FilpNormals : public Hitable {
 public:
  FilpNormals(Hitable *p_) : p(p_) {}

  virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
    if (p->hit(ray, t_min, t_max, rec)) {
      rec.n = -1 * rec.n;
      return true;
    } else {
      return false;
    }
  }

  virtual bool bounding_box(float t0, float t1, AABB &box) const {
    return p->bounding_box(t0, t1, box);
  }

  Hitable *p;
};


#endif  // !HITABLE_H__
