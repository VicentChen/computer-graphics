#ifndef SPHERE_H__
#define SPHERE_H__

#include "Hitable.h"
#include "Material.h"

inline void get_sphere_uv(const Vec3& p, float& u, float& v) {
  float phi = atan2(p.z(), p.x());
  float theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

class Sphere : public Hitable {
 public:
  Vec3 c;   // center
  float r;  // radius
  Material* mat_ptr;

  Sphere() {}
  Sphere(Vec3 c, float r, Material* mat_ptr) : c(c), r(r), mat_ptr(mat_ptr) {}
  ~Sphere() {}
  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const;
  virtual bool bounding_box(float t0, float t1, AABB& box) const;
};

// Solving equation, X is hit point, C is sphere center, R is radius:
// ------
//  (X-C)^2 = R^2
//=>(O+t*D-C)^2 = R^2
//=>(O-C)^2 + 2t(O-C)D + t^2*D^2 - R^2 = 0
//=>D^2*t^2 + 2(O-C)Dt + (O-C)^2-R^2 = 0
//  ^^^^(A)   ^^^^^^^(B) ^^^^^^^^^^^(C)
//
//  DELTA = B^2 - 4AC
//=>DELTA = 4 * ((B/2)^2 - AC)
//
//  x1 = (-B - sqrt(DELTA)) / (2 * A) = (-B/2 - sqrt(DELTA/4)) / A
//  x2 = (-B + sqrt(DELTA)) / (2 * A) = (-B/2 + sqrt(DELTA/4)) / A

bool Sphere::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  Vec3 CO = ray.o - c;
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
    get_sphere_uv((rec.p - c) / r, rec.u, rec.v);
    rec.n = (rec.p - c) / r;
    rec.mat_ptr = mat_ptr;
    return true;
  }
  return false;
}

bool Sphere::bounding_box(float t0, float t1, AABB& box) const {
  box = AABB(c - Vec3(r, r, r), c + Vec3(r, r, r));
  return true;
}

#endif  // !SPHERE_H__
