#ifndef MOVING_SPHERE_H__
#define MOVING_SPHERE_H__

#include "Sphere.h"

class MovingSphere : public Sphere{
 public:
  MovingSphere() {}
  MovingSphere(Vec3 c0_, Vec3 c1_, float t0_, float t1_, float r_, Material *mat_ptr_)
      : c0(c0_), c1(c1_), t0(t0_), t1(t1_), r(r_), mat_ptr(mat_ptr_) {}

  Vec3 center(float time) const {
    return c0 + (time - t0) / (t1 - t0) * (c1 - c0);
  }

  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const;

  Vec3 c0, c1;
  float t0, t1;
  float r;
  Material *mat_ptr;
};

bool MovingSphere::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  Vec3 CO = ray.o - center(ray.time());
  float A = dot(ray.d, ray.d);
  float B = dot(CO, ray.d);
  float C = dot(CO, CO) - r * r;
  float DET = B * B - A * C;  // delta

  if (DET > 0) {
    float x = (-B - sqrt(DET)) / A;
    if (x <= t_min || x >= t_max) {
      x = (-B + sqrt(DET)) / A;
      if (x <= t_min || x >= t_max) {
        return false;
      }
    }

    rec.t = x;
    rec.p = ray.point_at_parameter(x);
    rec.n = (rec.p - center(ray.time())) / r;
    rec.mat_ptr = mat_ptr;
    return true;
  }
  return false;
}

#endif  // !MOVING_SPHERE_H__
