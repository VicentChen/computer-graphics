#ifndef RECTANGLE_H__
#define RECTANGLE_H__

#include "AABB.h"
#include "Hitable.h"
#include "Material.h"

class XYRect : public Hitable {
 public:
  XYRect() {}
  XYRect(float x0_, float x1_, float y0_, float y1_, float k_, Material* mat_)
      : x0(x0_), y0(y0_), x1(x1_), y1(y1_), k(k_), mat(mat_) {}

  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
  virtual bool bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(Vec3(x0, y0, k - 0.0001), Vec3(x1, y1, k + 0.0001));
    return true;
  }

  Material* mat;
  float x0, x1, y0, y1, k;
};

bool XYRect::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  float t = (k - ray.o.z()) / ray.d.z();
  if (t < t_min || t > t_max) return false;

  float x = ray.o.x() + t * ray.d.x();
  float y = ray.o.y() + t * ray.d.y();

  if (x < x0 || x > x1 || y < y0 || y > y1) return false;

  rec.u = (x - x0) / (x1 - x0);
  rec.v = (y - y0) / (y1 - y0);
  rec.t = t;
  rec.mat_ptr = mat;
  rec.p = ray.point_at_parameter(t);
  rec.n = Vec3(0, 0, 1);
  return true;
}

class XZRect : public Hitable {
 public:
  XZRect() {}
  XZRect(float x0_, float x1_, float z0_, float z1_, float k_, Material* mat_)
      : x0(x0_), z0(z0_), x1(x1_), z1(z1_), k(k_), mat(mat_) {}

  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
  virtual bool bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(Vec3(x0, k - 0.0001, z0), Vec3(x1, k + 0.0001, z1));
    return true;
  }

  Material* mat;
  float x0, x1, z0, z1, k;
};

bool XZRect::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  float t = (k - ray.o.y()) / ray.d.y();
  if (t < t_min || t > t_max) return false;

  float x = ray.o.x() + t * ray.d.x();
  float z = ray.o.z() + t * ray.d.z();

  if (x < x0 || x > x1 || z < z0 || z > z1) return false;

  rec.u = (x - x0) / (x1 - x0);
  rec.v = (z - z0) / (z1 - z0);
  rec.t = t;
  rec.mat_ptr = mat;
  rec.p = ray.point_at_parameter(t);
  rec.n = Vec3(0, 1, 0);
  return true;
}

class YZRect : public Hitable {
 public:
  YZRect() {}
  YZRect(float y0_, float y1_, float z0_, float z1_, float k_, Material* mat_)
      : y0(y0_), z0(z0_), y1(y1_), z1(z1_), k(k_), mat(mat_) {}

  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
  virtual bool bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(Vec3(k - 0.0001, y0, z0), Vec3(k + 0.0001, y1, z1));
    return true;
  }

  Material* mat;
  float y0, y1, z0, z1, k;
};

bool YZRect::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  float t = (k - ray.o.x()) / ray.d.x();
  if (t < t_min || t > t_max) return false;

  float y = ray.o.y() + t * ray.d.y();
  float z = ray.o.z() + t * ray.d.z();

  if (y < y0 || y > y1 || z < z0 || z > z1) return false;

  rec.u = (y - y0) / (y1 - y0);
  rec.v = (z - z0) / (z1 - z0);
  rec.t = t;
  rec.mat_ptr = mat;
  rec.p = ray.point_at_parameter(t);
  rec.n = Vec3(1, 0, 0);
  return true;
}

#endif  // !RECTANGLE_H__
